unit downloadthreadmanagers;
{
  DownloadThreadmanagers keeps track of MAX_MANAGED_THREADS slots which can contain a running
  DownloadThread.

  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)

}
interface

uses SyncObjs, SysUtils,
     jobs, downloadthreads, stkconstants, identities, threadmanagers, loggers;


type TCommThreadManager = class(TThreadManager)
  public
    constructor Create(var logger : TLogger);
    destructor  Destroy();
    procedure setProxy(proxy, port : String);

  protected
    logger_ : TLogger;
    proxy_,
    port_   : String;
end;

type
  TDownloadThreadManager = class(TCommThreadManager)
  public
    constructor Create(var logger : TLogger);

    function download(url, targetPath, targetFilename : String): Longint;

    procedure updateStatus; virtual;

    procedure setMaxThreads(x: Longint);
    procedure clearFinishedThreads;
end;
  
implementation

constructor TCommThreadManager.Create(var logger : TLogger);
begin
  inherited Create(DEFAULT_DOWN_THREADS);
  logger_ := logger;
  proxy_ := '';
  port_ := '';
end;

destructor TCommThreadManager.Destroy();
begin
  inherited;
end;

procedure TCommThreadManager.setProxy(proxy, port : String);
begin
 CS_.Enter;
 proxy_ := proxy;
 port_ := port;
 CS_.Leave;
end;

constructor TDownloadThreadManager.Create(var logger : TLogger);
begin
 inherited Create(logger);
 updateStatus;
end;

function TDownloadThreadManager.download(url, targetPath, targetFilename : String): Longint;
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
              raise Exception.Create('Internal error in downloadthreadmanagers.pas, slot is -1');
            end;
  
  Inc(current_threads_);  
  slots_[slot] := TDownloadThread.Create(url, targetPath, targetFilename, proxy_, port_, logger_);
  updateStatus;

  Result := slot;
  CS_.Leave;
end;

procedure TDownloadThreadManager.updateStatus;
begin
  TMDownStatus.maxthreads := max_threads_;
  TMDownStatus.threads := current_threads_;
  TMDownStatus.isIdle  := isIdle();
  TMDownStatus.hasResources := hasResources();
end;

procedure TDownloadThreadManager.setMaxThreads(x: Longint);
begin
  inherited setMaxThreads(x);
  updateStatus();
end;

procedure TDownloadThreadManager.clearFinishedThreads;
begin
  inherited clearFinishedThreads;
  updateStatus();
end;

end.
  
