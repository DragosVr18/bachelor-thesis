import { useState } from "react";
import { infer } from "../service/inferService";
import { dataURLToBlob } from "../utils/dataURLToBlob";
import { API_URL } from "../config/api";

export function useInfer() {
    const [busy, setBusy]       = useState(false);
    const [overlayUrl, setOverlayUrl] = useState("");
    const [glbUrl, setGlbUrl]         = useState("");
    const [duration, setDuration]     = useState(null);

    const doInfer = async (input, isUpload, threshold, fov) => {
        setBusy(true);
        setOverlayUrl("");
        setGlbUrl("");
        setDuration(null);

        const t0 = performance.now();
        try {
            const fileBlob = isUpload
                ? input
                : dataURLToBlob(input);
            const filename = isUpload
                ? undefined
                : "webcam.png";

            const data = await infer(fileBlob, threshold, fov, filename);

            const t1 = performance.now();
            setDuration(t1 - t0);

            const stamp = `?t=${Date.now()}`;
            setOverlayUrl(`${API_URL}${data.overlay_url}${stamp}`);
            setGlbUrl(`${API_URL}${data.glb_url}${stamp}`);
        } catch (err) {
            console.error(err);
            alert(err.message);
        } finally {
            setBusy(false);
        }
    };

    return { busy, overlayUrl, glbUrl, duration, doInfer };
}
