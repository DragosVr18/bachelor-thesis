import React, { Suspense } from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { Stage } from "@react-three/drei";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { SmartControls } from "./SmartControls";

export function Model({ url }) {
    const gltf = useLoader(GLTFLoader, url);
    return <primitive object={gltf.scene} dispose={null} />;
}

export function ModelCanvas({ url }) {
    return (
        <Canvas shadows camera={{ position: [0, 0, 2], near: 0.1, far: 10 }} className="w-full h-[560px] border rounded shadow">
            <Stage>
                <Suspense fallback={null}>
                    <Model url={url} />
                </Suspense>
            </Stage>
            <SmartControls />
        </Canvas>
    );
}