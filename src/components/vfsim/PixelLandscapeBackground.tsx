/** Decorative fixed backdrop only; never overlays graph panels or catches clicks. */
export function PixelLandscapeBackground() {
  return (
    <div className="pixel-landscape" aria-hidden="true">
      <svg className="pixel-clouds" viewBox="0 0 1600 160" preserveAspectRatio="xMidYMin slice" focusable="false" shapeRendering="crispEdges">
        <path d="M34 66h24V52h18V38h43v14h19v14h23v16H34z M399 103h22V89h16V76h34v13h19v14h23v14H399z M1130 66h28V52h18V38h45v14h20v14h29v16h-140z M1460 113h21v-12h14V89h31v12h17v12h19v12h-102z" />
      </svg>
      <svg className="pixel-terrain" viewBox="0 0 1600 215" preserveAspectRatio="none" focusable="false" shapeRendering="crispEdges">
        {/* Low, stepped environmental silhouettes; no full-scene sun or mountains. */}
        <path className="pixel-terrain-far" d="M0 106h45V93h53V79h57v17h67V84h42V65h56v14h48v15h69V80h57v13h61V69h58v19h62v-9h49V63h63v17h64v14h67V78h52v16h63V73h58v17h59V69h64v21h71V79h57V63h53v18h59v13h63V76h57v17h81V75h58v25h48v115H0z" />
        <path className="pixel-terrain-mid" d="M0 146h57v-9h48v-21h61v17h60v-18h58v20h59v-15h61v18h76v-19h59v16h51v-14h68v20h60v-22h57v17h71v-17h62v21h66v-22h65v19h55v-23h62v26h70v-19h58v18h64v-24h65v23h62v-14h75v15h61V117h57v21h60v77H0z" />
        <path className="pixel-terrain-near" d="M0 172h64v-11h60v-15h52v18h65v-13h59v17h59v-22h74v15h65v-15h67v24h59v-17h54v15h57v-14h60v19h66v-26h58v17h70v-20h60v20h64v-17h69v20h60v-16h71v-19h65v22h72v-14h63v15h69v-17h57v21h66v-16h58v52H0z" />
        <path className="pixel-terrain-top" d="M0 188h80v-7h72v-5h75v8h89v-6h93v7h68v-7h64v9h100v-7h83v8h90v-5h67v7h92v-7h86v9h80v-7h95v9h99v-8h70v10h130v-8h67v27H0z" />
        <path className="pixel-terrain-ground" d="M0 201h144v-5h83v5h123v-7h67v7h170v-6h91v6h139v-8h79v8h181v-5h94v5h137v-7h90v7h102v14H0z" />
        <g className="pixel-terrain-plant">
          <path d="M90 197v-15h-6v-9h5v9h6v-15h5v15h6v-8h5v11h-11v12z M342 191v-14h-5v-7h5v9h6v-17h4v17h6v-9h5v11h-11v10z M841 198v-17h-6v-9h5v10h6v-14h6v13h5v-7h4v13h-9v11z M1393 199v-16h-7v-8h5v10h6v-18h5v17h6v-9h5v11h-11v13z" />
        </g>
      </svg>
    </div>
  );
}
