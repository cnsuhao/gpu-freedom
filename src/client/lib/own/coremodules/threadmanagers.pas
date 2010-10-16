unit threadmanagers;
{
  Threadmanagers keeps track of MAX_THREADS slots which can contain a running
  ComputationThread. A new ComputationThread can be created on a slot
  by using the Compute(...) method after defining a TJob structure.
  This class is the only class which can istantiate a new ComputationThread.

  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)

}
interface

uses SyncObjs, SysUtils,
     jobs, computationthreads, stkconstants, identities, pluginmanagers,
     methodcontrollers, resultcollectors, frontendmanagers;

type
  TThreadManager = class(TObject)
  public

    constructor Create(var plugman : TPluginManager; var meth : TMethodController;
                       var res : TResultCollector; var frontman : TFrontendManager);
    destructor  Destroy();
    
    // computes  a job if there are available threads
    // returns threadId if created, -1 if not
    function Compute(Job: TJob): Longint;
    
    // at regular intervals, this method needs to be called by the core
    procedure ClearFinishedThreads;
 
    // getters and setters
    // the number of threads can be changed dynamically
    procedure setMaxThreads(x: Longint);
    function  getMaxThreads() : Longint;
    function  isIdle() : Boolean;
    function  hasResources() : Boolean;
    function  getCurrentThreads() : Longint;

  private
    max_threads_, 
    current_threads_ : Longint;
    is_idle_         : Boolean;

    slots_        : Array[1..MAX_THREADS] of TComputationThread;

    CS_           : TCriticalSection;
    plugman_      : TPluginManager;
    meth_         : TMethodController;
    rescollector_ : TResultCollector;
    frontman_     : TFrontendManager;

    function findAvailableSlot() : Longint;
    procedure updateCoreIdentity;
  end;
  
implementation

constructor TThreadManager.Create(var plugman : TPluginManager; var meth : TMethodController;
                                  var res : TResultCollector; var frontman : TFrontendManager);
var i : Longint;
begin
  inherited Create();

  max_threads_ := DEFAULT_THREADS;
  current_threads_ := 0;

  CS_ := TCriticalSection.Create();
  for i:=1 to MAX_THREADS do slots_[i] := nil;

  plugman_      := plugman;
  meth_         := meth;
  rescollector_ := res;
  frontman_     := frontman;

  updateCoreIdentity;
end;

destructor TThreadManager.Destroy();
begin
  CS_.Free;
  inherited;
end;



function TThreadManager.findAvailableSlot() : Longint;
var i : Longint;
begin
  Result := -1;
  // we look for slots only until max_threads_ which can change dynamically
  for i:=1 to max_threads_ do 
    if slots_[i]=nil then
       begin
         Result := i;
         Exit;
       end;
end;

function TThreadManager.Compute(job: TJob): Longint;
var slot : Longint;
begin
  CS_.Enter;
  Result := -1;
  if not hasResources() then 
       begin
        Job.stack.error.ErrorID  := NO_AVAILABLE_THREADS_ID;
        Job.stack.error.ErrorMsg := NO_AVAILABLE_THREADS;
        Job.stack.error.ErrorArg := 'All slots ('+IntToStr(max_threads_)+') are full.';
        CS_.Leave;
        Exit;
       end; 
  
   slot := findAvailableSlot;
   if slot=-1 then
            begin
              CS_.Leave;
              raise Exception.Create('Internal error in threadmanagers.pas, slot is -1');
            end;
  
  Inc(current_threads_);  
  slots_[slot] := TComputationThread.Create(plugman_, meth_, rescollector_, frontman_, job, slot);
  updateCoreIdentity;

  Result := slot;
  CS_.Leave;
end;


procedure TThreadManager.ClearFinishedThreads;
var i : Longint;
begin
  // here we traverse the complete array
  // as the number of threads can change dynamically
  CS_.Enter;
  for i:=1 to MAX_THREADS do
    if (slots_[i] <> nil) and slots_[i].isJobDone() then
    begin
      slots_[i].WaitFor;
      FreeAndNil(slots_[i]);
      Dec(current_threads_);
    end;                             
  updateCoreIdentity;
  CS_.Leave;
end;

procedure TThreadManager.setMaxThreads(x: Longint);
begin
  CS_.Enter;
  max_threads_ := x;
  updateCoreIdentity;
  CS_.Leave;
end;

function  TThreadManager.getMaxThreads() : Longint;
begin
 Result := max_threads_;
end;

function  TThreadManager.isIdle() : Boolean;
begin
 Result := (current_threads_ = 0);
end;

function  TThreadManager.getCurrentThreads : Longint;
begin
 Result := current_threads_;
end;

function  TThreadManager.hasResources() : Boolean;
begin
   Result := (current_threads_<max_threads_);
end;

procedure TThreadManager.updateCoreIdentity;
begin
  myCoreId.maxthreads := max_threads_;
  myCoreId.threads := current_threads_;
  myCoreId.isIdle  := isIdle();
  myCoreId.hasResources := hasResources();
end;

end.
  
