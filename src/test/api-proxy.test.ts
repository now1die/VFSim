import { afterEach, describe, expect, it, vi } from "vitest";
import { proxyApi } from "@/lib/api-proxy.server";

const frontend = "https://frontend.example.test";
const backend = "https://backend.example.test";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("hosted API proxy", () => {
  it("forwards an empty session POST, relaying only the VFSim cookie", async () => {
    const session = "a".repeat(43);
    const cookie = `__Host-vfsim_session=${session}; Path=/; Secure; HttpOnly; SameSite=lax`;
    vi.stubEnv("VFSIM_API_URL", backend);
    vi.stubEnv("VFSIM_PROXY_TOKEN", "test-token-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    const fetchUpstream = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 204, headers: { "set-cookie": cookie } }));
    vi.stubGlobal("fetch", fetchUpstream);
    const url = new URL("/api/session", frontend);
    const request = new Request(url, {
      method: "POST",
      body: "",
      headers: { origin: frontend, cookie: `other=private; __Host-vfsim_session=${session}` },
    });

    const result = await proxyApi(request, url);

    expect(result.status).toBe(204);
    expect(result.headers.get("set-cookie")).toBe(cookie);
    expect(result.headers.get("cache-control")).toBe("no-store");
    expect(String(fetchUpstream.mock.calls[0]?.[0])).toBe(`${backend}/api/session`);
    const headers = new Headers(fetchUpstream.mock.calls[0]?.[1]?.headers);
    expect(headers.get("cookie")).toBe(`__Host-vfsim_session=${session}`);
    expect(headers.get("x-vfsim-proxy-token")).toBe("test-token-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  });

  it("rejects cross-origin writes before contacting the backend", async () => {
    const fetchUpstream = vi.fn();
    vi.stubGlobal("fetch", fetchUpstream);
    const url = new URL("/api/vfs/create", frontend);
    const request = new Request(url, {
      method: "POST",
      headers: { origin: "https://other.example.test" },
    });
    expect((await proxyApi(request, url)).status).toBe(403);
    expect(fetchUpstream).not.toHaveBeenCalled();
  });

  it("requires Origin even on an empty mutation", async () => {
    const fetchUpstream = vi.fn();
    vi.stubGlobal("fetch", fetchUpstream);
    const url = new URL("/api/session", frontend);
    expect((await proxyApi(new Request(url, { method: "POST" }), url)).status).toBe(403);
    expect(fetchUpstream).not.toHaveBeenCalled();
  });

  it("drops ambiguous session cookies and preserves throttling", async () => {
    vi.stubEnv("VERCEL", "1");
    vi.stubEnv("VFSIM_API_URL", backend);
    vi.stubEnv("VFSIM_PROXY_TOKEN", "test-token-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    const fetchUpstream = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        Response.json({ detail: "limit" }, { status: 429, headers: { "retry-after": "120" } }),
      );
    vi.stubGlobal("fetch", fetchUpstream);
    const url = new URL("/api/session", frontend);
    const cookie = `__Host-vfsim_session=${"a".repeat(43)}`;
    const result = await proxyApi(
      new Request(url, {
        method: "POST",
        headers: {
          origin: frontend,
          cookie: `${cookie}; ${cookie}`,
          "x-forwarded-for": "192.0.2.1",
          "x-vfsim-client-ip": "192.0.2.2",
        },
      }),
      url,
    );
    const forwarded = new Headers(fetchUpstream.mock.calls[0]?.[1]?.headers);
    expect(forwarded.has("cookie")).toBe(false);
    expect(forwarded.get("x-vfsim-client-ip")).toBe("192.0.2.1");
    expect(result.status).toBe(429);
    expect(result.headers.get("retry-after")).toBe("120");
  });

  it("requires an HTTPS upstream in Vercel", async () => {
    vi.stubEnv("VERCEL", "1");
    vi.stubEnv("VFSIM_API_URL", "http://127.0.0.1:8000");
    vi.stubEnv("VFSIM_PROXY_TOKEN", "test-token-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    const url = new URL("/api/health", frontend);
    expect((await proxyApi(new Request(url), url)).status).toBe(503);
  });
});
