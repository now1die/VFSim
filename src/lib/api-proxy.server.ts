import { isIP } from "node:net";

const maxBodyBytes = 16_384;
const sessionCookie = /^__Host-vfsim_session=[A-Za-z0-9_-]{43}$/;
const proxyToken = /^[A-Za-z0-9_-]{32,}$/;

function failure(status: number, detail: string) {
  return Response.json({ detail }, { status, headers: { "cache-control": "no-store" } });
}

export async function proxyApi(request: Request, url: URL) {
  if (!["GET", "POST", "DELETE"].includes(request.method)) {
    return failure(405, "Unsupported API method.");
  }
  if (request.method !== "GET") {
    const origin = request.headers.get("origin");
    if (request.headers.get("sec-fetch-site") === "cross-site" || origin !== url.origin) {
      return failure(403, "API writes must come from VFSim.");
    }
  }

  const configuredUrl = process.env["VFSIM_API_URL"];
  const token = process.env["VFSIM_PROXY_TOKEN"];
  if (!configuredUrl || !token || !proxyToken.test(token)) {
    return failure(503, "The VFSim backend is not configured.");
  }
  let upstream: URL;
  try {
    upstream = new URL(configuredUrl);
    const localHttp =
      process.env["VERCEL"] !== "1" &&
      upstream.protocol === "http:" &&
      ["localhost", "127.0.0.1", "[::1]"].includes(upstream.hostname);
    if (
      (!localHttp && upstream.protocol !== "https:") ||
      upstream.username ||
      upstream.password ||
      upstream.pathname !== "/" ||
      upstream.search ||
      upstream.hash
    ) {
      throw new Error("Expected a backend origin");
    }
  } catch {
    return failure(503, "The VFSim backend configuration is invalid.");
  }

  upstream.pathname = url.pathname;
  upstream.search = url.search;
  const headers = new Headers({ accept: "application/json", "x-vfsim-proxy-token": token });
  // Forward only VFSim's cookie. Vercel and site cookies belong to the frontend.
  const cookies = (request.headers.get("cookie") ?? "").split(";").map((cookie) => cookie.trim());
  const sessions = cookies.filter((cookie) => cookie.startsWith("__Host-vfsim_session="));
  const [session] = sessions;
  if (session && sessions.length === 1 && sessionCookie.test(session))
    headers.set("cookie", session);
  // Vercel overwrites X-Forwarded-For at ingress. Never forward a client-provided
  // origin key or trust forwarding headers on a local preview.
  if (process.env["VERCEL"] === "1" && url.pathname === "/api/session") {
    const address = request.headers.get("x-forwarded-for")?.trim();
    if (address && isIP(address)) headers.set("x-vfsim-client-ip", address);
  }

  const timeout = AbortSignal.timeout(25_000);
  const signal = AbortSignal.any([request.signal, timeout]);
  try {
    let body: Uint8Array<ArrayBuffer> | undefined;
    if (request.body) {
      const reader = request.body.getReader();
      const buffer = new Uint8Array(maxBodyBytes);
      let size = 0;
      const cancel = () => void reader.cancel().catch(() => undefined);
      signal.addEventListener("abort", cancel, { once: true });
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (size + value.byteLength > maxBodyBytes) {
            cancel();
            return failure(413, "The request is too large.");
          }
          buffer.set(value, size);
          size += value.byteLength;
        }
      } finally {
        signal.removeEventListener("abort", cancel);
        reader.releaseLock();
      }
      if (size > 0) {
        if (request.headers.get("content-type")?.split(";")[0] !== "application/json") {
          return failure(415, "Expected a JSON request.");
        }
        body = buffer.subarray(0, size);
        headers.set("content-type", "application/json");
      }
    }

    const result = await fetch(upstream, {
      method: request.method,
      headers,
      ...(body ? { body } : {}),
      redirect: "error",
      signal,
    });
    if (result.status >= 500) {
      await result.body?.cancel();
      return failure(result.status, "The VFSim backend is temporarily unavailable.");
    }
    const contentType = result.headers.get("content-type");
    if (result.status !== 204 && !contentType?.startsWith("application/json")) {
      await result.body?.cancel();
      return failure(502, "The VFSim backend returned an unexpected response.");
    }
    const responseHeaders = new Headers({ "cache-control": "no-store" });
    if (contentType) responseHeaders.set("content-type", contentType);
    const retryAfter = result.headers.get("retry-after");
    if (result.status === 429 && retryAfter && /^\d{1,6}$/.test(retryAfter)) {
      responseHeaders.set("retry-after", retryAfter);
    }
    const cookie = result.headers.get("set-cookie");
    const cookiePair = cookie?.split(";")[0];
    if (
      url.pathname === "/api/session" &&
      result.status === 204 &&
      cookie &&
      cookiePair &&
      sessionCookie.test(cookiePair)
    ) {
      responseHeaders.set("set-cookie", cookie);
    }
    return new Response(result.body, { status: result.status, headers: responseHeaders });
  } catch {
    return timeout.aborted
      ? failure(504, "The VFSim backend took too long to respond.")
      : failure(502, "The VFSim backend could not be reached.");
  }
}
