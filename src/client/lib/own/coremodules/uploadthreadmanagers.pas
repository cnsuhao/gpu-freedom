unit uploadthreadmanagers;
{
  UploadThreadmanagers keeps track of MAX_MANAGED_THREADS slots which can contain a running
  UploadThread.

  (c) by 2002-2011 the GPU Development Team
  (c) by 2011 HB9TVM
  This unit is released under GNU Public License (GPL)

}
interface

uses SyncObjs, SysUtils,
     jobs, uploadthreads, stkconstants, identities, downloadthreadmanagers, loggers;

type
  TUploadThreadManager = class(TCommThreadManager)
  public
    constructor Create(var logger : TLogger);

    function upload(url, sourcePath, sourceFilename : String): Longint;

    procedure updateStatus; virtual;

    procedure setMaxThreads(x: Longint);
    procedure clearFinishedThreads;
  end;

implementation

constructor TUploadThreadManager.Create(var logger : TLogger);
begin
 inherited Create(logger);
 updateStatus;
end;


function TUploadThreadManager.upload(url, sourcePath, sourceFilename : String): Longint;
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
              raise Exception.Create('Internal error in uploadthreadmanagers.pas, slot is -1');
            end;

  Inc(current_threads_);
  slots_[slot] := TUploadThread.Create(url, sourcePath, sourceFilename, proxy_, port_, logger_);
  updateStatus;

  Result := slot;
  CS_.Leave;
end;

procedure TUploadThreadManager.updateStatus;
begin
  TMUploadStatus.maxthreads := max_threads_;
  TMUploadStatus.threads := current_threads_;
  TMUploadStatus.isIdle  := isIdle();
  TMUploadStatus.hasResources := hasResources();
end;

procedure TUploadThreadManager.setMaxThreads(x: Longint);
begin
  inherited setMaxThreads(x);
  updateStatus();
end;

procedure TUploadThreadManager.clearFinishedThreads;
begin
  inherited clearFinishedThreads;
  updateStatus();
end;

end.
