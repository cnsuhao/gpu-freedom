unit threadmanagers;
{
  Threadmanagers keeps track of MAX_MANAGED_THREADS slots which can contain a running
  ManagedThread.

  (c) by 2002-2010 the GPU Development Team
  (c) by 2010 HB9TVM
  This unit is released under GNU Public License (GPL)

}
interface

uses SyncObjs, SysUtils,
     jobs, computationthreads, stkconstants, identities, pluginmanagers,
     methodcontrollers, resultcollectors, frontendmanagers,
     managedthreads, loggers;

type
  TThreadManager = class(TObject)
  public

    constructor Create(maxThreads : Longint);
    constructor Create(maxThreads : Longint; var logger : TLogger);
    destructor  Destroy();

    // at regular intervals, this method needs to be called by the core
    procedure clearFinishedThreads;

    // sets status of thread manager
    procedure updateStatus; virtual; abstract;

    // getters and setters
    // the number of threads can be changed dynamically
    procedure setMaxThreads(x: Longint);
    function  getMaxThreads() : Longint;
    function  isIdle() : Boolean;
    function  hasResources() : Boolean;
    function  getCurrentThreads() : Longint;

    procedure printThreadStatus(name : String; var logger : TLogger);


  protected
    max_threads_, 
    current_threads_ : Longint;
    is_idle_         : Boolean;

    slots_        : Array[1..MAX_MANAGED_THREADS] of TManagedThread;
    names_        : Array[1..MAX_MANAGED_THREADS] of String;

    CS_           : TCriticalSection;
    logger_       : TLogger;

    function findAvailableSlot() : Longint;

  end;



type TCommThreadManager = class(TThreadManager)
  public
    constructor Create(var logger : TLogger);
    destructor  Destroy();
    procedure setProxy(proxy, port : String);

  protected
    proxy_,
    port_   : String;
end;
  
implementation

constructor TThreadManager.Create(maxThreads : Longint);
var i : Longint;
begin
  inherited Create();

  if maxThreads>MAX_MANAGED_THREADS then
     raise Exception.Create('Internal error in threadmanagers.pas');

  max_threads_ := maxThreads;
  current_threads_ := 0;

  CS_ := TCriticalSection.Create();
  for i:=1 to MAX_MANAGED_THREADS do slots_[i] := nil;

end;

constructor TThreadManager.Create(maxThreads : Longint; var logger : TLogger);
begin
  Create(maxThreads);
  logger_ := logger;
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


procedure TThreadManager.clearFinishedThreads;
var i : Longint;
begin
  // here we traverse the complete array
  // as the number of threads can change dynamically
  CS_.Enter;
  for i:=1 to MAX_MANAGED_THREADS do
    if (slots_[i] <> nil) and slots_[i].isDone() then
    begin
      slots_[i].WaitFor;
      FreeAndNil(slots_[i]);
      Dec(current_threads_);
    end;                             
  CS_.Leave;
end;

procedure TThreadManager.setMaxThreads(x: Longint);
begin
  CS_.Enter;
  max_threads_ := x;
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

procedure TThreadManager.printThreadStatus(name : String; var logger : TLogger);
var i   : Longint;
    msg : String;
begin
  logger.log(LVL_DEBUG, '********************************');
  logger.log(LVL_DEBUG, name+' status');
  logger.log(LVL_DEBUG, 'Current threads: '+IntToSTr(current_threads_));
  logger.log(LVL_DEBUG, 'Max     threads: '+IntToSTr(max_threads_));

  for i:=1 to current_threads_ do
     begin
      if Assigned(slots_[i]) then
               begin
                 msg := '';
                 if slots_[i].isErroneus() then msg := msg+'ERROR ' else msg := msg+'OK    ';
                 if slots_[i].isDone() then msg := msg+'Done.      ' else msg := msg+'Running... ';
               end
      else msg := 'nil         ';
      logger.log(IntToStr(i)+': '+msg+' '+names_[i]);
     end; // for

  logger.log(LVL_DEBUG, '********************************');

end;


constructor TCommThreadManager.Create(var logger : TLogger);
begin
  inherited Create(DEFAULT_DOWNLOAD_THREADS, logger);
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


end.
  
