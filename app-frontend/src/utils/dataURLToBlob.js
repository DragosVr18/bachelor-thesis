export function dataURLToBlob(dataURL) {
    const [header, b64] = dataURL.split(",");
    const mime = header.match(/:(.*?);/)[1];
    const binary = atob(b64);
    const u8 = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) u8[i] = binary.charCodeAt(i);
    return new Blob([u8], { type: mime });
}