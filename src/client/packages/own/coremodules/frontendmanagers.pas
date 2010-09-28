unit frontendmanagers;
{
 This units manages the registering of jobIds
 which need to be notified back to the frontends.
 
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
    
}
interface

uses SyncObjs, SysUtils,
     stkconstants, stacks;

const
  QUEUE_SIZE = 120;

type 
  // How frontends will be notified of the result
  // ct_UDP_IP:  frontend is using UDP/IP channels, we store its UDP port (e.g. 32145)
  //             and IP, mostly localhost as IP
  // ct_None:    dummy if answer is not needed, e.g. if GPU core itself registers in the
  //             queue
  // ct_Files:   frontends wait for an answer in a particular directory of the filesystem
  TContactType = (ct_None, ct_UDP_IP, ct_Files);
  
type
  TRegisterInfo = record
        jobID,               // jobID the frontend sent to GPU Core
	                     // used for lookup
	
	IP : String;         // if contact type is ct_Port_IP
	port : Longint;
    
	path,
	filename : String;   // if contact type is ct_Files
	
	executable,
	formname,
        fullname	: String;   // information on process name and form name
	
        typeID:  TContactType;
  end;  

type TRegisterQueue = class(TObject)
   public    
	  constructor Create();
	  destructor Destroy();
	  
	  procedure registerJob(var reg : TRegisterInfo);
	  procedure unregisterJob(jobID : String; formName : String);
      function findRI4Job(jobId : String; var reg : TRegisterInfo) : Boolean;
	  function findMultipleRI4Job(jobId : String; var reg : TRegisterInfo; var start : Longint) : Boolean;
	  function getRegisteredList(var stk : TStack) : Boolean;
        
    private
          queue_ : Array [1..QUEUE_SIZE] of TRegisterInfo;
	  idx_   : Longint; // index on queue_
	  CS_    : TCriticalSection;
	  
	  procedure initQueueCell(i : Longint);
end;

type TFrontendManager = class(TObject)
   public
          constructor Create();
	  destructor Destroy;
	  
	  function getStandardQueue() : TRegisterQueue;
	  function getBroadcastQueue() : TRegisterQueue;
	  
	  function prepareRegisterInfo4Core(jobId : String) : TRegisterInfo;
	  function prepareRegisterInfo4UdpFrontend(jobId, IP : String; port : Longint; executable, form, fullname : String) : TRegisterInfo;
	  function prepareRegisterInfo4FileFrontend(jobId, path, filename : String; executable, form, fullname : String) : TRegisterInfo;
	  
   private
          standardQ_  : TRegisterQueue;
	  broadcastQ_ : TRegisterQueue;
end;

implementation

constructor TRegisterQueue.Create();
var i : Longint;
begin
  inherited;
  CS_ := TCriticalSection.Create();
  for i:=1 to QUEUE_SIZE do
    initQueueCell(i);
end;

destructor TRegisterQueue.Destroy();
begin
  CS_.Free;
  inherited;
end;

procedure TRegisterQueue.initQueueCell(i : Longint);
begin
  queue_[i].jobID := '';
  queue_[i].IP := '';
  queue_[i].port := 0;
  queue_[i].path := '';
  queue_[i].filename := '';
  queue_[i].executable := '';
  queue_[i].formname := '';
  queue_[i].fullname := '';
  queue_[i].typeId := ct_None;
end;

procedure TRegisterQueue.registerJob(var reg : TRegisterInfo);
begin
  Inc(idx_);
  if (idx_>QUEUE_SIZE) then idx_ := 1;

  queue_[idx_].jobID := reg.jobId;
  queue_[idx_].IP := reg.IP;
  queue_[idx_].port := reg.port;
  queue_[idx_].path := reg.path;
  queue_[idx_].filename := reg.filename;
  queue_[idx_].executable := reg.executable;
  queue_[idx_].formname := reg.formname;
  queue_[idx_].fullname := reg.fullname;
  queue_[idx_].typeId := reg.typeId;
end;

procedure TRegisterQueue.unregisterJob(jobID : String; formName : String);
var i : Longint;
begin
  CS_.Enter;
  for i:=1 to QUEUE_SIZE do
     if (queue_[i].jobId=jobId) and (queue_[i].formName=formName) then
	    begin
		  initQueueCell(i);
		  CS_.Leave;
		  Exit;
		end;
  
  CS_.Leave;
end;

function TRegisterQueue.findRI4Job(jobId : String; var reg : TRegisterInfo) : Boolean;
var start : Longint;
begin
  start  := 1;
  Result := findMultipleRI4Job(jobId, reg, start);
end;

function TRegisterQueue.findMultipleRI4Job(jobId : String; var reg : TRegisterInfo; var start : Longint) : Boolean;
var i : Longint;
begin
  Result := false;
  if (start<1) then Exit;
  if (start>QUEUE_SIZE) then raise Exception.Create('findMultipleRI4Job called with start argument outside QUEUE_SIZE');
  if (Trim(jobId)='') then raise Exception.Create('jobId was empty in findMultipleRI4Job');
  CS_.Enter;
  for i:=start to QUEUE_SIZE do
     if (queue_[i].jobId = jobId) then
	         begin
		   reg := queue_[i];
                   Result := true;
		   CS_.Leave;
                   Exit;
		 end;   
  CS_.Leave;  
end; 

function TRegisterQueue.getRegisteredList(var stk : TStack) : Boolean;
var i : Longint;
begin
  Result := true;
  CS_.Enter;
  for i:=1 to QUEUE_SIZE do
     if (queue_[i].jobId <>'') and Result then
	    begin
		  Result := pushStr(queue_[i].jobId, stk);
		  Result := pushStr(queue_[i].formname, stk);
		  Result := pushStr(queue_[i].fullname, stk);
		end;
  
  CS_.Leave;
end;      

constructor TFrontendManager.Create();
begin
  inherited;
  standardQ_ := TRegisterQueue.Create();
  broadcastQ_ := TRegisterQueue.Create();
end;

destructor TFrontendManager.Destroy;
begin
  standardQ_.Free;
  broadcastQ_.Free;
  inherited;
end;

function TFrontendManager.getStandardQueue() : TRegisterQueue;
begin
 Result := standardQ_;
end;

function TFrontendManager.getBroadcastQueue() : TRegisterQueue;
begin
 Result := broadcastQ_;
end;

function TFrontendManager.prepareRegisterInfo4Core(jobId : String) : TRegisterInfo;
begin
  Result.jobID := jobId;
  Result.IP := '';
  Result.port := 0;
  Result.path := '';
  Result.filename := '';
  Result.executable := 'gpucore.exe';
  Result.formname := 'TGPUCore';
  Result.fullname := 'GPU core v'+GPU_CORE_VERSION;
  Result.typeId := ct_None;
end;

function TFrontendManager.prepareRegisterInfo4UdpFrontend(jobId, IP : String; port : Longint; executable, form, fullname : String) : TRegisterInfo;
begin
  Result.jobID := jobId;
  Result.IP := IP;
  Result.port := port;
  Result.path := '';
  Result.filename := '';
  Result.executable := executable;
  Result.formname := form;
  Result.fullname := fullname;
  Result.typeId := ct_UDP_IP;
end;

function TFrontendManager.prepareRegisterInfo4FileFrontend(jobId, path, filename : String; executable, form, fullname : String) : TRegisterInfo;
begin
  Result.jobID := jobId;
  Result.IP := '';
  Result.port := 0;
  Result.path := path;
  Result.filename := filename;
  Result.executable := executable;
  Result.formname := form;
  Result.fullname := fullname;
  Result.typeId := ct_Files;
end;

end.

