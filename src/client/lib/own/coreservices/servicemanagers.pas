unit servicemanagers;
 {
   TServiceThreadManager handles all services available to the GPU core.servicemanagers
   and is a TThreadManager descendant

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses
  threadmanagers, coreservices, identities, Sysutils;

type TServiceThreadManager = class(TThreadManager)
   public
    constructor Create(maxThreads : Longint);
    destructor Destroy;

    function launch(serviceThread : TCoreServiceThread): Longint;

    procedure setMaxThreads(x: Longint);
    procedure clearFinishedThreads;
    procedure updateStatus; virtual;
end;

implementation

constructor TServiceThreadManager.Create(maxThreads : Longint);
begin
  inherited Create(maxThreads);
end;

destructor TServiceThreadManager.Destroy;
begin
end;


function TServiceThreadManager.launch(serviceThread : TCoreServiceThread): Longint;
var slot : Longint;
begin
  CS_.Enter;
  Result := -1;
  if not hasResources() then
       begin
        CS_.Leave;
        Exit;
       end;

   slot := findAvailableSlot;
   if slot=-1 then
            begin
              CS_.Leave;
              raise Exception.Create('Internal error in servicemanagers.pas, slot is -1');
            end;

  Inc(current_threads_);
  slots_[slot] := serviceThread;
  updateStatus;

  Result := slot;
  serviceThread.Resume();
  CS_.Leave;
end;

procedure TServiceThreadManager.updateStatus;
begin
  tmServiceStatus.maxthreads := max_threads_;
  tmServiceStatus.threads := current_threads_;
  tmServiceStatus.isIdle  := isIdle();
  tmServiceStatus.hasResources := hasResources();
end;

procedure TServiceThreadManager.setMaxThreads(x: Longint);
begin
  inherited setMaxThreads(x);
  updateStatus();
end;

procedure TServiceThreadManager.clearFinishedThreads;
begin
  inherited clearFinishedThreads;
  updateStatus();
end;
end.
