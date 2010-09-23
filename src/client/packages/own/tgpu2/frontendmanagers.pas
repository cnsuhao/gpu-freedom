unit frontendmanagers;

interface

uses gpuconstants;

const
  QUEUE_SIZE = 120;

type 
  // How frontends will be notified of the result
  // ct_Port_IP: frontend is using UDP/IP channels, we store its UDP port (e.g. 32145)
  //             and IP, mostly localhost as IP
  // ct_None:    dummy if answer is not needed, e.g. if GPU core itself registers in the
  //             queue
  // ct_Files:   frontends wait for an answer in a particular directory of the filesystem
  TContactType = (ct_None, ct_Port, ct_Files);
  
type
  TRegisterInfo = record
    jobID,               // jobID the frontend sent to GPU Core
	IP : String;         // if contact type is ct_Port_IP
	port : Longint;
    
	path,
	filename : String;   // if contact type is ct_Files
	
    typeID:  TContactType;
  end;  

type TRegisterQueue = class(TObject)
   public    
	  constructor Create();
	  destructor Destroy();
  
  
    private
      queue_ : Array [1..QUEUE_SIZE] of TRegisterInfo;
	  
	  function initQueueCell(i : Longint);
end;

type TFrontendManager = class(TObject);
   public
   
   private
      standardQ  : TRegisterQueue;
	  broadcastQ : TRegisterQueue;
end;

implementation

constructor TRegisterQueue.Create();
begin
end;

destructor TRegisterQueue.Destroy();
begin
end;

function TRegisterQueue.initQueueCell(i : Longint);
begin
end;

end.

