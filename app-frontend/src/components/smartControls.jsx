import React, { useRef, useState, useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

export function SmartControls() {
    const controls = useRef();
    const { gl } = useThree();
    const [inside, setInside] = useState(false);

    useEffect(() => {
        const enter = () => setInside(true);
        const leave = () => setInside(false);
        const el = gl.domElement;
        el.addEventListener("pointerenter", enter);
        el.addEventListener("pointerleave", leave);
        return () => {
            el.removeEventListener("pointerenter", enter);
            el.removeEventListener("pointerleave", leave);
        };
    }, [gl]);

    useFrame(() => {
        if (controls.current) controls.current.enabled = inside;
    });

    return (
        <OrbitControls
            ref={controls}
            makeDefault
            enableDamping
            minDistance={1}
            maxDistance={5}
        />
    );
}