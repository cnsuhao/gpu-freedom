unit compservicemanagers;
 {
   TCompServiceThreadManager handles all computational services for the GPU core.
   compservicemanager is a TThreadManager descendant

   (c) by 2010-2013 HB9TVM and the GPU Team
}

interface

uses
  threadmanagers, coreservices, computationservices, identities, loggers, Sysutils;

type TCompServiceThreadManager = class(TThreadManager)
   public
    constructor Create(maxThreads : Longint; var logger : TLogger);
    destructor Destroy;

    function launch(var compThread : TComputationServiceThread): Longint;

    procedure setMaxThreads(x: Longint);
    procedure clearFinishedThreads;
    procedure updateStatus; virtual;
end;

implementation

constructor TCompServiceThreadManager.Create(maxThreads : Longint; var logger : TLogger);
begin
  inherited Create(maxThreads, logger);
end;

destructor TCompServiceThreadManager.Destroy;
begin
end;


function TCompServiceThreadManager.launch(var compThread : TComputationServiceThread): Longint;
var slot : Longint;
begin
  CS_.Enter;
  Result := -1;
  if not hasResources() then
       begin
        logger_.log(LVL_WARNING, 'No resources in TCompServiceThreadManager, releasing service.');
        if Assigned(compThread) then compThread.Free;
        CS_.Leave;
        Exit;
       end;

   slot := findAvailableSlot;
   if slot=-1 then
            begin
              CS_.Leave;
              if Assigned(compThread) then compThread.Free;
              raise Exception.Create('Internal error in servicemanagers.pas, slot is -1');
            end;

  Inc(current_threads_);
  compThread.setSlot(slot);
  slots_[slot] := compThread;
  names_[slot] := 'TComputationServiceThread';
  updateStatus;

  Result := slot;
  compThread.Resume();
  CS_.Leave;
end;

procedure TCompServiceThreadManager.updateStatus;
begin
  tmCompStatus.maxthreads := max_threads_;
  tmCompStatus.threads := current_threads_;
  tmCompStatus.isIdle  := isIdle();
  tmCompStatus.hasResources := hasResources();
end;

procedure TCompServiceThreadManager.setMaxThreads(x: Longint);
begin
  inherited setMaxThreads(x);
  updateStatus();
end;

procedure TCompServiceThreadManager.clearFinishedThreads;
begin
  inherited clearFinishedThreads;
  updateStatus();
end;
end.
