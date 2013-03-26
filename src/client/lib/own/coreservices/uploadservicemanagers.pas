unit uploadservicemanagers;
{
  TUploadServiceManager keeps track of MAX_MANAGED_THREADS slots which can contain a running
  db aware TUploadServiceThread.

  (c) by 2002-2013 the GPU Development Team and HB9TVM
  This unit is released under GNU Public License (GPL)
}
interface

uses SyncObjs, SysUtils,
     jobs, uploadservices, stkconstants, identities, threadmanagers, loggers;


type
  TUploadServiceManager = class(TThreadManager)
  public
    constructor Create(maxThreads : Longint; var logger : TLogger);

    function launch(var upThread : TUploadServiceThread): Longint;

    procedure updateStatus; virtual;

    procedure setMaxThreads(x: Longint);
    procedure clearFinishedThreads;
end;
  
implementation


constructor TUploadServiceManager.Create(maxThreads : Longint; var logger : TLogger);
begin
 inherited Create(maxThreads, logger);
 updateStatus;
end;

function TUploadServiceManager.launch(var upThread : TUploadServiceThread): Longint;
var slot : Longint;
begin
  CS_.Enter;
  Result := -1;
  if not hasResources() then
         begin
          logger_.log(LVL_WARNING, 'No resources in TUploadServiceManager, releasing service.');
          CS_.Leave;
          if Assigned(upThread) then upThread.Free;
          Exit;
         end;

     slot := findAvailableSlot;
     if slot=-1 then
              begin
                CS_.Leave;
                if Assigned(upThread) then upThread.Free;
                raise Exception.Create('Internal error in uploadservicemanagers.pas, slot is -1');
              end;

    Inc(current_threads_);
    slots_[slot] :=  upThread;
    names_[slot] := 'TUploadServiceThread';
    updateStatus;

    Result := slot;
    upThread.Resume();
    CS_.Leave;
end;

procedure TUploadServiceManager.updateStatus;
begin
  tmUploadStatus.maxthreads   := max_threads_;
  tmUploadStatus.threads      := current_threads_;
  tmUploadStatus.isIdle       := isIdle();
  tmUploadStatus.hasResources := hasResources();
end;

procedure TUploadServiceManager.setMaxThreads(x: Longint);
begin
  inherited setMaxThreads(x);
  updateStatus();
end;

procedure TUploadServiceManager.clearFinishedThreads;
begin
  inherited clearFinishedThreads;
  updateStatus();
end;

end.
  
