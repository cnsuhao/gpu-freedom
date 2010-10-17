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
     managedthreads;

type
  TThreadManager = class(TObject)
  public

    constructor Create(maxThreads : Longint);
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

  protected
    max_threads_, 
    current_threads_ : Longint;
    is_idle_         : Boolean;

    slots_        : Array[1..MAX_MANAGED_THREADS] of TManagedThread;

    CS_           : TCriticalSection;

    function findAvailableSlot() : Longint;

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
  updateStatus;
  CS_.Leave;
end;

procedure TThreadManager.setMaxThreads(x: Longint);
begin
  CS_.Enter;
  max_threads_ := x;
  updateStatus;
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


end.
  
