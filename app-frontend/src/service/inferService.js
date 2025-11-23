import { API_URL } from "../config/api";

export async function infer(fileBlob, threshold, fov, filename) {
    const fd = new FormData();
    if (filename) {
        fd.append("file", fileBlob, filename);
    } else {
        fd.append("file", fileBlob);
    }
    fd.append("threshold", threshold.toString());
    fd.append("fov", fov.toString());

    const res = await fetch(`${API_URL}/infer`, {
        method: "POST",
        body: fd,
        mode: "cors",
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}
