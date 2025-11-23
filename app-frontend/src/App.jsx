import React, { useRef, useState, useEffect } from "react";
import { ModeSwitch } from "./components/ModeSwitch";
import { InputCard } from "./components/InputCard";
import { ModelCanvas } from "./components/Model";
import { useInfer } from "./hooks/useInfer";
import "./App.css";

export default function App() {
    const [source, setSource] = useState("upload");
    const [threshold, setThreshold] = useState(0.3);
    const [fov, setFov] = useState(60);
    const [timerSec, setTimerSec] = useState(3);
    const [uploadPreview, setUploadPreview] = useState(null);
    const [countdown, setCountdown] = useState(0);

    const camRef = useRef(null);
    const fileInputRef = useRef(null);

    const { busy, overlayUrl, glbUrl, duration, doInfer } = useInfer();

    useEffect(() => {
        return () => {
            if (uploadPreview) URL.revokeObjectURL(uploadPreview);
        };
    }, [uploadPreview]);

    const handleInfer = () => {
        if (source === "upload") {
            const file = fileInputRef.current?.files?.[0];
            if (!file) return alert("No file chosen");
            doInfer(file, true, threshold, fov).then(() => {
                fileInputRef.current.value = "";
                setUploadPreview(null);
            });
        }
        else {
            if (countdown > 0) return;
            setCountdown(timerSec);
            let cd = timerSec;
            const iv = setInterval(() => {
                cd -= 1;
                setCountdown(cd);
                if (cd === 0) {
                    clearInterval(iv);
                    const shot = camRef.current?.getScreenshot();
                    if (!shot) return alert("No webcam image");
                    doInfer(shot, false, threshold, fov);
                }
            }, 1000);
        }
    };

    return (
        <div className="flex flex-col items-center gap-14 w-full max-w-screen-xl px-6 py-10">
            <h1 className="text-4xl font-bold text-center">Mini-HMR</h1>

            {duration != null && (
                <p className="text-center text-base text-gray-500">
                    Inference took {(duration / 1000).toFixed(2)} s
                </p>
            )}

            <ModeSwitch
                mode={source}
                onChange={(v) => {
                    setSource(v);
                    setUploadPreview(null);
                    setCountdown(0);
                }}
            />

            <InputCard
                source={source}
                camRef={camRef}
                fileInputRef={fileInputRef}
                uploadPreview={uploadPreview}
                setUploadPreview={setUploadPreview}
                threshold={threshold}
                setThreshold={setThreshold}
                fov={fov}
                setFov={setFov}
                timerSec={timerSec}
                setTimerSec={setTimerSec}
                countdown={countdown}
                busy={busy}
                onInfer={handleInfer}
            />

            {overlayUrl && (
                <div className="grid lg:grid-cols-2 gap-10 items-start w-[70rem] justify-items-center">
                    <img
                        src={overlayUrl}
                        alt="overlay"
                        className="w-full aspect-video object-contain border rounded shadow"
                    />
                    <ModelCanvas url={glbUrl} />
                </div>
            )}
        </div>
    );
}
