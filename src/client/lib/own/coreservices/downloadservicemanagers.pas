unit downloadservicemanagers;
{
  TDownloadServiceManager keeps track of MAX_MANAGED_THREADS slots which can contain a running
  db aware TDownloadServiceThread.

  (c) by 2002-2013 the GPU Development Team and HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses SyncObjs, SysUtils,
     jobs, downloadservices, stkconstants, identities, threadmanagers, loggers;


type
  TDownloadServiceManager = class(TThreadManager)
  public
    constructor Create(maxThreads : Longint; var logger : TLogger);

    function launch(var downThread : TDownloadServiceThread): Longint;

    procedure updateStatus; virtual;

    procedure setMaxThreads(x: Longint);
    procedure clearFinishedThreads;
end;
  
implementation


constructor TDownloadServiceManager.Create(maxThreads : Longint; var logger : TLogger);
begin
 inherited Create(maxThreads, logger);
 updateStatus;
end;

function TDownloadServiceManager.launch(var downThread : TDownloadServiceThread): Longint;
var slot : Longint;
begin
  CS_.Enter;
  Result := -1;
  if not hasResources() then
         begin
          logger_.log(LVL_WARNING, 'No resources in TDownloadServiceManager, releasing service.');
          if Assigned(downThread) then downThread.Free;
          CS_.Leave;
          Exit;
         end;

     slot := findAvailableSlot;
     if slot=-1 then
              begin
                CS_.Leave;
                raise Exception.Create('Internal error in downloadservicemanagers.pas, slot is -1');
              end;

    Inc(current_threads_);
    slots_[slot] :=  downThread;
    names_[slot] := 'TDownloadServiceThread';
    updateStatus;

    Result := slot;
    downThread.Resume();
    CS_.Leave;
end;

procedure TDownloadServiceManager.updateStatus;
begin
  tmDownStatus.maxthreads := max_threads_;
  tmDownStatus.threads := current_threads_;
  tmDownStatus.isIdle  := isIdle();
  tmDownStatus.hasResources := hasResources();
end;

procedure TDownloadServiceManager.setMaxThreads(x: Longint);
begin
  inherited setMaxThreads(x);
  updateStatus();
end;

procedure TDownloadServiceManager.clearFinishedThreads;
begin
  inherited clearFinishedThreads;
  updateStatus();
end;

end.
  
