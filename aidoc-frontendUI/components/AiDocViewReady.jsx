import React, { useState, useRef } from 'react';

const imgVector = "http://localhost:3845/assets/755ab1d13ae46a1d6ff8bd93fc22064a189d6b9c.svg";
const imgVector1 = "http://localhost:3845/assets/599a4214c75e380369fda145961759ce49c42cde.svg";
const imgGroup = "http://localhost:3845/assets/70726a64c15f285a6322592983956eb8d102b273.svg";
const imgGroup1 = "http://localhost:3845/assets/31a926cd2101af44ef7ac222667c17de515e848b.svg";
const imgCamerabuttonWithsymbol = "http://localhost:3845/assets/78df247c5dfc5f46068dce0de064710356cba5c1.svg";
const imgAmbulanceframe = "http://localhost:3845/assets/b7f04a043612189ef7e2337f1cc74af64ef7936a.svg";
const imgMic = "http://localhost:3845/assets/0cf3eb66da2cb44bea636979e990c018b233c2e1.svg";
const imgEllipse4 = "http://localhost:3845/assets/514f5a69fce530c034013ed0e9e604efc65441ca.svg";
const imgGroup2 = "http://localhost:3845/assets/3070f965639f3dd9e2a496dd6ec3ce964ddd830c.svg";


function ImageGallery() {
  return (
    <div className="relative size-full" data-name="image_gallery" data-node-id="17:35">
      <div className="absolute inset-[8.333%]" data-name="Group" data-node-id="17:32">
        <div className="absolute inset-[-1.64%]">
          <img alt className="block max-w-none size-full" src={imgGroup1} />
        </div>
      </div>
    </div>
  );
}

function MedicoSymbol() {
  return (
    <div className="relative size-full" data-name="medico_symbol" data-node-id="4:1043">
      <div className="absolute inset-[6.21%_1%_1.14%_1%]" data-name="Vector" data-node-id="4:1041">
        <img alt className="block max-w-none size-full" src={imgVector} />
      </div>
      <div className="absolute inset-[1.15%_39.84%_50.23%_39.82%]" data-name="Vector" data-node-id="4:1042">
        <img alt className="block max-w-none size-full" src={imgVector1} />
      </div>
    </div>
  );
}

function AmbulanceSymbol() {
  return (
    <div className="relative size-full" data-name="ambulance symbol" data-node-id="4:1071">
      <div className="absolute inset-[9.79%_7.5%_9.79%_5%]" data-name="Group" data-node-id="4:1061">
        <img alt className="block max-w-none size-full" src={imgGroup} />
      </div>
    </div>
  );
}

export default function AiDocViewReady() {
  const [transcript, setTranscript] = useState("");
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);

  const handleMicClick = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      alert('Speech recognition is not supported in this browser.');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!recognitionRef.current) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = false;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event) => {
        const result = event.results[0][0].transcript;
        setTranscript(result);
        setIsListening(false);
      };

      recognitionRef.current.onerror = () => {
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }

    if (!isListening) {
      setIsListening(true);
      recognitionRef.current.start();
    } else {
      setIsListening(false);
      recognitionRef.current.stop();
    }
  };

  return (
    <div className="bg-[#0a101c] relative size-full" data-name="AiDoc View - Ready" data-node-id="4:1035">
      <div className="absolute font-['Inter:Regular',_sans-serif] font-normal leading-[0] not-italic text-[#8a8a8e] text-[16px] text-nowrap top-[734px]" data-node-id="4:1036" style={{ left: "calc(50% - 136px)" }}>
        <p className="leading-[1.4] whitespace-pre">VIDEO</p>
      </div>
      <div className="absolute font-['Inter:Regular',_sans-serif] font-normal leading-[0] not-italic text-[16px] text-nowrap text-white top-[734px]" data-node-id="4:1037" style={{ left: "calc(50% + 90px)" }}>
        <p className="leading-[1.4] whitespace-pre">AiDOC</p>
      </div>
      <div className="absolute font-['Inter:Regular',_sans-serif] font-normal leading-[0] not-italic text-[#8a8a8e] text-[16px] text-nowrap top-[734px]" data-node-id="4:1038" style={{ left: "calc(50% - 28px)" }}>
        <p className="leading-[1.4] whitespace-pre">PHOTO</p>
      </div>
      <div className="absolute left-1/2 size-[90px] top-[767px] translate-x-[-50%]" data-name="camerabutton(withsymbol)" data-node-id="4:1039">
        <img alt className="block max-w-none size-full" src={imgCamerabuttonWithsymbol} />
      </div>
      <div className="absolute left-[316px] size-[50px] translate-y-[-50%]" data-name="ambulanceframe" data-node-id="4:1056" style={{ top: "calc(50% - 349px)" }}>
        <img alt className="block max-w-none size-full" src={imgAmbulanceframe} />
      </div>
      <div className="absolute contents left-[35px] top-[659px]" data-node-id="18:29">
        <div className="absolute bg-[rgba(255,255,255,0.1)] h-[41px] left-[35px] overflow-clip rounded-[25px] shadow-[0px_4px_25px_0px_rgba(0,0,0,0.25)] top-[659px] w-[307px]" data-name="chatwindow" data-node-id="4:1053">
          <div className="absolute font-['Inter:Regular',_sans-serif] font-normal h-[11px] leading-[0] not-italic text-[#8a8a8e] text-[16px] w-[124px]" data-node-id="5:8" style={{ top: "calc(50% - 10.5px)", left: "calc(50% - 117.5px)" }}>
            <p className="leading-[1.4]">{transcript || 'Type here....'}</p>
          </div>
          <div 
            className="absolute h-[22px] translate-x-[-50%] translate-y-[-50%] w-[18px] cursor-pointer hover:opacity-80" 
            data-name="mic" 
            data-node-id="16:82" 
            style={{ 
              top: "calc(50% + 0.5px)", 
              left: "calc(50% + 112.5px)",
              opacity: isListening ? 0.5 : 1 
            }}
            onClick={handleMicClick}
            title={isListening ? "Click to stop" : "Click to speak"}
          >
            <img alt className="block max-w-none size-full" src={imgMic} />
          </div>
        </div>
      </div>
      <div className="absolute bg-[rgba(255,255,255,0.1)] h-[50px] left-6 overflow-clip rounded-[25px] shadow-[0px_4px_25px_0px_rgba(0,0,0,0.25)] top-[63px] w-[170px]" data-name="severityframe" data-node-id="5:5">
        <div className="absolute left-[125px] size-[25px] top-3" data-node-id="5:11">
          <img alt className="block max-w-none size-full" src={imgEllipse4} />
        </div>
        <div className="absolute font-['Inter:Regular',_sans-serif] font-normal leading-[0] not-italic text-[#8a8a8e] text-[20px] text-nowrap top-[9px]" data-node-id="5:10" style={{ left: "calc(50% - 56px)" }}>
          <p className="leading-[1.4] whitespace-pre">Severity</p>
        </div>
      </div>
      <div className="absolute left-1/2 size-[62px] top-[784px] translate-x-[-50%]" data-name="medico_symbol" data-node-id="4:1043">
        <MedicoSymbol />
      </div>
      <div className="absolute h-[33px] left-80 overflow-clip top-[71px] w-[39px]" data-name="ambulance symbol" data-node-id="4:1071">
        <AmbulanceSymbol />
      </div>
      <div className="absolute left-[61px] size-[55px] top-[788px]" data-name="image_gallery" data-node-id="17:36">
        <ImageGallery />
      </div>
      <div className="absolute inset-[90.39%_15.42%_3.89%_72.14%]" data-name="Group" data-node-id="18:21">
        <div className="absolute inset-[-1.5%]">
          <img alt className="block max-w-none size-full" src={imgGroup2} />
        </div>
      </div>
    </div>
  );
}
