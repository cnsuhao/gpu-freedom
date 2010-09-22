{
  TGPU2 class is the main class which is the core of the core of GPU itself.

  It  keeps track of MAX_THREADS slots which can contain a running
  ComputationThread. A new ComputationThread can be created on a slot
  by using the Compute(...) method after defining a TJob structure. 
  This class is the only class which can istantiate a new ComputationThread.
  
  TGPU2 component encapsulates the PluginManager which manages Plugins
  which contain the computational algorithms.
  
  It encapsulates the MethodController, which makes sure that the same function
  inside a plugin is not called concurrently, for increased stability.
  
  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)
    
}
unit TGPU2;

interface

uses
  jobs, computationthreads, stacks, tgpuconstants, SyncObjs;

type
  TGPUCore2 = class(TObject)
  public

    constructor Create(var plugman : TPluginManager; var meth : TMethodController; var resColl : TResultCollector);
    destructor  Destroy();
    
    // computes  a job if there are available threads
    function Compute(Job: TJob): boolean;
    
    // at regular intervals, this method needs to be called by the core
    procedure ClearFinishedThreads;
 
    // getters and setters
    // the number of threads can be changed dynamically
    procedure setMaxThreads(x: Longint);
    function  getMaxThreads() : Longint;
    function  isIdle() : Boolean;
    function  hasResources() : Boolean;
    function  getCurrentThreads() : Longint;
	
	// helper structures
	function getPluginManger()   : TPluginManager;
	function getMethController() : TMethodController;
	function getSpecCommands()   : TSpecialCommand;
    function getResultCollector(): TResultCollector;
    
  private
    max_threads_, 
	   threads_running : Longint;
    is_idle_        : Boolean;
    plugman_        : TPluginManager;
    methController_ : TMethodController;
    speccommands_   : TSpecialCommand;
    rescoll_        : TResultCollector;
    // TODO: define a ResultCollector
    CS_             : TCriticalSection;
    
    slots_        : Array[1..MAX_THREADS] of TComputationThread;
    
    function findAvailableSlot() : Longint;
  end;
  
implementation

constructor TGPUCore2.Create(var plugman : TPluginManager; var meth : TMethodController; var resColl : TResultCollector);
var i : Longint;
begin
  inherited Create();
  plugman_ := plugman;
  methController_ := meth;
  rescoll_ := resColl;
  max_threads_ := DEFAULT_THREADS;
  current_threads_ := 0;
  CS_ := TCriticalSection.Create();
  speccommands_ := TSpecialCommand.Create(self);
  for i:=1 to MAX_THREADS do slots_[i] := nil;
end;

destructor TGPUCore2.Destroy();
begin
  CS_.Free;
  inherited;
end;

function TGPUCore2.getPluginManger()   : TPluginManager;
begin
 Result := plugman_;
end;

function TGPUCore2.getMethController() : TMethodController;
begin
 Result := methController_;
end;

function TGPUCore2.getSpecCommands() : TSpecialCommand;
begin
 Result := speccommands_;
end;

function getResultCollector(): TResultCollector;
begin
 Result := rescoll_;
end; 


function TGPUCore2.findAvailableSlot() : Longint;
var i : Longint;
begin
  Result := -1;
  // we look for slots only until max_threads_ which can change dynamically
  for i:=1 to max_threads_ do 
    if slots[i]=nil then
       begin
         Result := i;
         Exit;
       end;
end;

function TGPUCore2.Compute(job: TJob): boolean;
var slot : Longint;
begin
  CS_.Enter;
  Result := false;
  if not hasResources() then 
       begin
        Job.ErrorID  := NO_AVAILABLE_THREADS_ID;
        Job.ErrorMsg := NO_AVAILABLE_THREADS;
        Job.ErrorArg := 'All slots ('+IntToStr(max_threads_)+') are full.';
        CS_.Leave;
        Exit;
       end; 
  
   slot := findAvailableSlot;
   if slot=-1 then
            begin
              CS_.Leave;
              throw new Exception.Create('Internal error in tgpu2.pas, slot is -1');
            end;
  
  Inc(current_threads_);  
  slots_[slot] := TComputationThread.Create(self, job, slot); 
  
  Return := true;
  CS_.Leave;
end;


procedure TGPUCore2.ClearFinishedThreads;
var i : Longint;
begin
  // here we traverse the complete array
  // as the number of threads can change dynamically
  for i:=1 to MAX_THREADS do 
    if (slots_[i] <> nil) and slots_[i].isJobDone() then
    begin
      slots_[i].WaitFor;
      slots_[i].JobForThread.Free;
      FreeAndNil(slots_[i]);
      Dec(current_threads_);
    end;                             

end;

procedure TGPUCore2.setMaxThreads(x: Longint);
begin
  CS_.Enter;
  max_threads_ := x;
  CS_.Leave;
end;

function  TGPUCore2.getMaxThreads() : Longint;
begin
 Result := max_threads_;
end;

function  TGPUCore2.isIdle() : Boolean;
begin
 Result := (current_threads_ = 0);
end;

function  TGPUCore2.getCurrentThreads : Longint;
begin
 Result := current_threads_;
end;

function  TGPUCore2.hasResources() : Boolean;
begin
   Result := (current_threads_<max_threads_);
end;

end;
  