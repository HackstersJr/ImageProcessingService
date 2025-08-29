import { useState } from "react";
import CameraView from "./components/CameraView";
import AiDocViewReady from "./components/AiDocViewReady";

export default function App() {
  const [screen, setScreen] = useState("camera");
  return screen === "camera" ? (
    <CameraView onSwitchScreen={() => setScreen("aidoc")} />
  ) : (
    <AiDocViewReady onSwitchScreen={() => setScreen("camera")} />
  );
}