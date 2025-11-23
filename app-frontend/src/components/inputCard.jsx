import React, { useEffect } from "react";
import Webcam from "react-webcam";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";

export function InputCard({
                              source,
                              camRef,
                              fileInputRef,
                              uploadPreview,
                              setUploadPreview,
                              threshold,
                              setThreshold,
                              fov,
                              setFov,
                              timerSec,
                              setTimerSec,
                              countdown,
                              busy,
                              onInfer,
                          }) {
    useEffect(() => {
        return () => {
            if (uploadPreview) URL.revokeObjectURL(uploadPreview);
        };
    }, [uploadPreview]);

    return (
        <Card className="w-[70rem] min-w-xl text-lg">
            <CardContent className="p-8 flex flex-col gap-8">
                {source === "upload" ? (
                    <>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="image/*"
                            className="rounded text-base"
                            onChange={(e) => {
                                const f = e.target.files?.[0];
                                f ? setUploadPreview(URL.createObjectURL(f)) : setUploadPreview(null);
                            }}
                        />
                        {uploadPreview && (
                            <img
                                src={uploadPreview}
                                alt="selected"
                                className="max-w-sm mx-auto rounded shadow"
                            />
                        )}
                    </>
                ) : (
                    <div className="relative w-full max-w-md mx-auto">
                        <Webcam
                            ref={camRef}
                            audio={false}
                            screenshotFormat="image/png"
                            className="rounded-lg w-full"
                        />
                        {countdown > 0 && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-lg">
                <span className="text-7xl font-bold text-white drop-shadow-lg">
                  {countdown}
                </span>
                            </div>
                        )}
                    </div>
                )}

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
                    <label className="flex flex-col gap-1">
                        <span className="flex justify-between text-base font-medium">
                          <span>Detection Threshold</span>
                          <span className="font-mono">{threshold.toFixed(2)}</span>
                        </span>
                        <Slider
                            min={0.1}
                            max={0.7}
                            step={0.05}
                            value={[threshold]}
                            onValueChange={(v) => setThreshold(v[0])}
                        />
                        <div className="flex justify-between text-sm text-gray-500 mt-1 px-0.5">
                            {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].map((t) => (
                                <span key={t}>{t}</span>
                            ))}
                        </div>
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-base font-medium">Field of View (°)</span>
                        <select
                            className="border rounded p-3 bg-white text-base"
                            value={fov}
                            onChange={(e) => setFov(Number(e.target.value))}
                        >
                            {[50, 60, 70].map((v) => (
                                <option key={v} value={v}>
                                    {v}
                                </option>
                            ))}
                        </select>
                    </label>

                    {source === "webcam" && (
                        <label className="flex flex-col gap-1 sm:col-span-2">
                          <span className="flex justify-between text-base font-medium">
                            <span>Capture Timer (s)</span>
                            <span className="font-mono">{timerSec}</span>
                          </span>
                            <Slider
                                min={0}
                                max={10}
                                step={1}
                                value={[timerSec]}
                                onValueChange={(v) => setTimerSec(v[0])}
                            />
                        </label>
                    )}
                </div>

                <Button
                    disabled={busy || countdown > 0}
                    onClick={onInfer}
                    className="self-start px-6 py-3 text-lg font-semibold"
                >
                    {busy
                        ? "Running…"
                        : source === "webcam"
                            ? countdown === 0
                                ? "Capture"
                                : "…"
                            : "Infer"}
                </Button>
            </CardContent>
        </Card>
    );
}