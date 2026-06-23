import { useEffect, useRef } from "react";

/**
 * Waveform — the signature visual element of CARVIS.
 *
 * Renders an oscilloscope-style trace that mimics a live accelerometer
 * feed. Calm, narrow-amplitude waves for "owner" state; jagged,
 * wide-amplitude spikes for "intruder" state. Purely decorative/ambient
 * (not plotting real numeric data point-for-point) but driven by the
 * actual std values returned from the backend, so the visual intensity
 * is grounded in the real prediction.
 */
export default function Waveform({ state = "idle", intensity = 0.3, height = 120 }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const phaseRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let width = canvas.clientWidth;
    let h = canvas.clientHeight;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const color =
      state === "owner" ? "#3DDC84" : state === "intruder" ? "#FF4757" : "#565D6B";

    const baseAmplitude = state === "idle" ? 6 : 10 + intensity * 38;
    const jaggedness = state === "intruder" ? 1 : state === "owner" ? 0.15 : 0.3;
    const speed = state === "intruder" ? 0.18 : 0.05;

    function draw() {
      ctx.clearRect(0, 0, width, h);

      // subtle grid
      ctx.strokeStyle = "rgba(255,255,255,0.03)";
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 32) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.shadowBlur = 12;
      ctx.shadowColor = color;

      const midY = h / 2;
      const points = 140;

      for (let i = 0; i <= points; i++) {
        const x = (i / points) * width;
        const t = i * 0.18 + phaseRef.current;

        let y =
          Math.sin(t) * baseAmplitude +
          Math.sin(t * 2.7) * baseAmplitude * 0.3 * jaggedness +
          (Math.random() - 0.5) * baseAmplitude * jaggedness * 0.8;

        y = midY + y;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      phaseRef.current += speed;
      animRef.current = requestAnimationFrame(draw);
    }

    draw();

    const handleResize = () => {
      width = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = width * dpr;
      canvas.height = h * dpr;
      ctx.scale(dpr, dpr);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", handleResize);
    };
  }, [state, intensity]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: `${height}px`, display: "block" }}
    />
  );
}
