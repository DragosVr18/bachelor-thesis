import React from "react";

export function ModeSwitch({ mode, onChange }) {
    return (
        <div className="flex gap-4 justify-center">
            {["upload", "webcam"].map((v) => (
                <React.Fragment key={v}>
                    <input
                        type="radio"
                        id={`mode-${v}`}
                        value={v}
                        checked={mode === v}
                        onChange={() => onChange(v)}
                        className="rg-item"
                    />
                    <label htmlFor={`mode-${v}`} className="rg-label capitalize">
                        {v}
                    </label>
                </React.Fragment>
            ))}
        </div>
    );
}