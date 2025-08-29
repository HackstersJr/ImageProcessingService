const imgEllipse1 = "http://localhost:3845/assets/82bdc8c15d6d38e0f1ac172a5c6ccdb7fea58d0b.svg";

export default function CameraView({ onSwitchScreen }) {
  return (
    <div className="bg-[#0a101c] relative size-full" data-name="camera view" data-node-id="4:3">
      <div className="absolute font-['Inter:Regular',_sans-serif] font-normal leading-[0] not-italic text-[#8a8a8e] text-[16px] text-nowrap top-[734px]" data-node-id="4:5" style={{ left: "calc(50% - 133px)" }}>
        <p className="leading-[1.4] whitespace-pre">VIDEO</p>
      </div>
      <div className="absolute font-['Inter:Regular',_sans-serif] font-normal leading-[0] not-italic text-[#ffd60a] text-[16px] text-nowrap top-[734px] cursor-pointer" data-node-id="4:1030" style={{ left: "calc(50% + 90px)" }} onClick={onSwitchScreen}>
        <p className="leading-[1.4] whitespace-pre">AiDOC</p>
      </div>
      <div className="absolute font-['Inter:Regular',_sans-serif] font-normal leading-[0] not-italic text-[16px] text-nowrap text-white top-[734px]" data-node-id="4:6" style={{ left: "calc(50% - 25px)" }}>
        <p className="leading-[1.4] whitespace-pre">PHOTO</p>
      </div>
      <div className="absolute left-[169px] size-[70px] top-[774px]" data-node-id="4:1034">
        <img alt className="block max-w-none size-full" src={imgEllipse1} />
      </div>
    </div>
  );
}
