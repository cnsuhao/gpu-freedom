unit servicemanagers;
 {
   TServiceThreadManager handles all services available to the GPU core.servicemanagers
   and is a TThreadManager descendant

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses
  threadmanagers, coreservices, identities, loggers, Sysutils;

type TServiceThreadManager = class(TThreadManager)
   public
    constructor Create(maxThreads : Longint);
    constructor Create(maxThreads : Longint; var logger : TLogger);
    destructor Destroy;

    function launch(var serviceThread : TCoreServiceThread; threadname : String): Longint;

    procedure setMaxThreads(x: Longint);
    procedure clearFinishedThreads;
    procedure updateStatus; virtual;

end;

implementation

constructor TServiceThreadManager.Create(maxThreads : Longint);
begin
  inherited Create(maxThreads);
end;


constructor TServiceThreadManager.Create(maxThreads : Longint; var logger : TLogger);
begin
  inherited Create(maxThreads, logger);
end;

destructor TServiceThreadManager.Destroy;
begin
end;


function TServiceThreadManager.launch(var serviceThread : TCoreServiceThread; threadname : String): Longint;
var slot : Longint;
begin
  CS_.Enter;
  Result := -1;
  if not hasResources() then
       begin
        logger_.log(LVL_WARNING, 'No resources in TServiceThreadManager, releasing service.');
        //TODO: check if freeing inside launch does not create memory leak
        if Assigned(serviceThread) then serviceThread.Free;
        CS_.Leave;
        Exit;
       end;

   slot := findAvailableSlot;
   if slot=-1 then
            begin
              CS_.Leave;
              //TODO: check if freeing inside launch does not create memory leak
              if Assigned(serviceThread) then serviceThread.Free;
              raise Exception.Create('Internal error in servicemanagers.pas, slot is -1');
            end;

  Inc(current_threads_);
  slots_[slot] := serviceThread;
  names_[slot] := threadName;
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
