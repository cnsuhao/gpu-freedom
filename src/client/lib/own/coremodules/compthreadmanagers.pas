unit compthreadmanagers;
{
  CompThreadmanagers keeps track of MAX_MANAGED_THREADS slots which can contain a running
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
     methodcontrollers, resultcollectors, frontendmanagers, threadmanagers;

type
  TCompThreadManager = class(TThreadManager)
  public

    constructor Create(var plugman : TPluginManager; var meth : TMethodController;
                       var res : TResultCollector; var frontman : TFrontendManager);
    destructor  Destroy();
    
    // computes  a job if there are available threads
    // returns threadId if created, -1 if not
    function Compute(Job: TJob): Longint;

    procedure updateStatus; virtual;

    procedure setMaxThreads(x: Longint);
    procedure clearFinishedThreads;

  private
    plugman_      : TPluginManager;
    meth_         : TMethodController;
    rescollector_ : TResultCollector;
    frontman_     : TFrontendManager;

  end;
  
implementation

constructor TCompThreadManager.Create(var plugman : TPluginManager; var meth : TMethodController;
                                  var res : TResultCollector; var frontman : TFrontendManager);
begin
  inherited Create(DEFAULT_COMP_THREADS);

  plugman_      := plugman;
  meth_         := meth;
  rescollector_ := res;
  frontman_     := frontman;

  updateStatus;
end;

destructor TCompThreadManager.Destroy();
begin
  inherited;
end;


function TCompThreadManager.Compute(job: TJob): Longint;
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
  updateStatus;

  Result := slot;
  CS_.Leave;
end;

procedure TCompThreadManager.updateStatus;
begin
  TMCompStatus.maxthreads := max_threads_;
  TMCompStatus.threads := current_threads_;
  TMCompStatus.isIdle  := isIdle();
  TMCompStatus.hasResources := hasResources();
end;

procedure TCompThreadManager.setMaxThreads(x: Longint);
begin
  inherited setMaxThreads(x);
  updateStatus();
end;

procedure TCompThreadManager.clearFinishedThreads;
begin
  inherited clearFinishedThreads;
  updateStatus();
end;

end.
  
