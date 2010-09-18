{
  TGPU_component class is the main class which registers the GPU component on 
  the Delphi bar.

  It  keeps track of MAX_THREADS slots which can contain a running
  ComputationThread. A new ComputationThread can be created on a slot
  by using the Compute(...) method after defining a TJob structure. 
  This class is the only class which can istantiate a new ComputationThread.
  
  Results are collected into the JobsCollected array. However, this mechanism
  is only for internal purposes. Frontends are still responsible to collect results
  in a reliable way.
  
  TGPU_component encapsulates the PluginManager which manages Plugins
  which contain the computational algorithms.
  
  It encapsulates the FunctionCallController, which makes sure that function
  calls inside plugins are not done concurrently.
  
  It encapsulates FrontQueue and BroadCastFrontQueue, which are queues
  of jobs to be executed together with the information of which frontend
  sent the job. (TODO: move the entire logic inside the GPU component
  and not only the queues). BroadCastFrontQueue is used for frontends
  which register to a particular service (like CHATBOT_BROADCAST
  or XML_NETMAPSTATS_BROADCAST).
  
}
unit TGPU_component;

interface

uses
  SysUtils, Classes, PluginManager, ComputationThread, Jobs, Definitions,
  common, gpu_utils, FrontendManager, FunctionCallController;

const
  MAX_THREADS = 3;  {maximum number of allowed threads}

type
  TGPU = class(TComponent)
  private
    FormatSet: TFormatSet;
    procedure SetMaxThreads(x: integer);
  protected
    { Protected declarations }
    FMaxThreads, FCurrentThreads: integer;
    FIdle:     boolean;
    procedure OnTerminatedThread(Sender: TObject);
  public
    { Public declarations }
    // list of jobs we keep also sum, average, min, max}
    // index of JobsCollected corresponds to the job id}
    JobsCollected: array [1..MAX_COLLECTING_IDS] of TGPUCollectResult;

    // list of running threads
    WorkThread: array [1..MAX_THREADS] of TComputationThread;

    //Function Call Controller is used to avoid having two
    //threads into the same DLL function
    FuncController : TFunctionCallController;

    // the plugin manager, responsible to load, discard
    // and search functions into DLLs
    PlugMan: TPluginManager;

    // FrontQueue is the circular list where jobs sent by frontends
    // are remembered. Here we store a tuple of records,
    // from which port (UDP Channel)
    // or how the frontend name was (Windows Message Channel)
    // together with job identifier
    // this information is needed by GPU core (gpu_mainform.pas)
    // to send back results to the frontends.
    FrontQueue,
    // BroadcastServiceFrontQueue is like FrontQueue, but here
    // we store frontends registering to special services like Chatbot
    // channels or network status information (for Netmapper)
    BroadcastServiceFrontQueue: TFrontendManager;

    procedure AfterConstruction; override;
    procedure BeforeDestruction; override;

    function Compute(Job: TJobOnGnutella): boolean;
    procedure ClearFinishedThreads;
    procedure ClearAllThreads;
    procedure CollectResults(JobId, Result: string; ComputedTime: TDateTime);
    procedure InitGPUResults;
    function GetNextFreeSlot: integer;
    procedure SetUpdate(Update: boolean);

  published
    { Published declarations }
    property MaxThreads: integer Read FMaxThreads Write SetMaxThreads;
    property CurrentThreads: integer Read FCurrentThreads;
    property Idle: boolean Read FIdle Write FIdle;
  end;


procedure Register;

implementation


procedure Register;
begin
  RegisterComponents('GPU', [TGPU]);
end;

procedure TGPU.SetMaxThreads(x: integer);
begin
  if x < 1 then
    FMaxThreads := 1
  else if x > MAX_THREADS then
    FMaxThreads := MAX_THREADS
  else
    FMaxThreads := x;
end;

procedure TGPU.AfterConstruction;
begin
  FCurrentThreads := 0;
  InitGPUResults;

  PlugMan    := TPluginManager.Create;
  FrontQueue := TFrontendManager.Create;
  BroadcastServiceFrontQueue := TFrontendManager.Create;
  FormatSet  := TFormatSet.Create;
  FIdle      := True;
  FuncController := TFunctionCallController.Create(MAX_THREADS);
  // we allow Terragen jobs to run concurrently, because at the time
  // Terragen jobs were implemented like that
  FuncController.allowRunningFunctionConcurrently('terragengrid');
end;

procedure TGPU.BeforeDestruction;
begin
  ClearAllThreads;
  PlugMan.Free;
  FrontQueue.Free;
  BroadcastServiceFrontQueue.Free;
  FuncController.Free;
  FormatSet.Free;
end;

{returns true, if the GPU is not busy and accepts new threads}
procedure TGPU.ClearFinishedThreads;
var
  i: integer;
begin
  for i := 1 to FMaxThreads do
    if (WorkThread[i] <> nil) and WorkThread[i].JobDone then
    begin
      WorkThread[i].WaitFor;
      WorkThread[i].JobForThread.Free;
      FreeAndNil(WorkThread[i]);
      Dec(FCurrentThreads);
    end;


  if FCurrentThreads < FMaxThreads then
    FIdle := True
  else
    FIdle := False;
end;

procedure TGPU.ClearAllThreads;
var
  i: integer;
begin
  for i := 1 to MAX_THREADS do
    if WorkThread[i] <> nil then
    begin
      //TODO: reenable it
      //if not WorkThread[i].JobDone then
      //  TerminateThread(WorkThread[i].Handle, 0);
      FreeAndNil(WorkThread[i]);
    end;

  FCurrentThreads := 0;
  FIdle := True;

  //we also clear the critical section management
  FuncController.Clear;
end;


{returns false, if all threads are busy}
function TGPU.Compute(Job: TJobOnGnutella): boolean;
var
  i: integer;
  ThreadCreated: boolean;
begin
  ThreadCreated := False;
  i := 1;
  repeat
    if WorkThread[i] = nil then
    begin
      ThreadCreated := True;
      WorkThread[i] := TComputationThread.Create(True);
      Job.ComputedTime := Time;

      {passing parameters to the thread}
      WorkThread[i].JobForThread := Job;
      WorkThread[i].PlugMan      := PlugMan;
      WorkThread[i].UpdateNeeded := True;
      WorkThread[i].FuncController := FuncController;
      WorkThread[i].FuncThreadID := i;
      WorkThread[i].Resume; {this wakes the thread up}
      Inc(FCurrentThreads);
    end;
    Inc(i);
  until (i > FMaxThreads) or ThreadCreated;

  Result := True;
end;

procedure TGPU.InitGPUResults;
var
  i: integer;
begin
  {initialize collecting results}
  for i := 1 to MAX_COLLECTING_IDS do
  begin
    JobsCollected[i].FirstResult := 0;
    JobsCollected[i].LastResult := 0;
    JobsCollected[i].N      := 0;
    JobsCollected[i].Sum    := 0;
    JobsCollected[i].Min    := 1E250;
    JobsCollected[i].Max    := -1E250;
    JobsCollected[i].StdDev := 0;
  end;
end;


procedure TGPU.CollectResults(JobId, Result: string; ComputedTime: TDateTime);
var
  i:   integer;
  res: extended;
  arg : String;
begin
  Result := ExtractParam(Result, ',');
  if (Result = WRONG_GPU_COMMAND) then
    Exit;
  if not isFloat(Result) then
    Exit;
  try
    i := StrToInt(JobID);
    begin
      Arg := StringReplace (Result, '.', DecimalSeparator, []);
      Arg := StringReplace (Arg, ',', DecimalSeparator, []);

      res := StrToFloat(Arg);
    end;

    if ((i > 0) and (i <= MAX_COLLECTING_IDS)) then
    begin
      with JobsCollected[i] do
      begin
        LastResult := res;
        TotalTime  := TotalTime + ComputedTime;
        if N = 0 then
          FirstResult := res;
        Inc(N);
        if res < Min then
          Min := res;
        if res > Max then
          Max := res;
        Sum := Sum + res;
        Avg    := Sum / N;
        SumStdDev := SumStdDev + Sqr(res - Avg);
        StdDev := Sqrt(SumStdDev / N);
      end; {with}
    end;
  except
  end;
end;


procedure TGPU.OnTerminatedThread(Sender: TObject);
begin
end;

function TGPU.GetNextFreeSlot: integer;
var
  i: integer;
begin
  i := 0;

  repeat
    Inc(i);
    if (WorkThread[i] = nil) then
    begin
      Result := i;
      Exit;
    end;

  until i >= FMaxThreads;

  Result := -1;
end;

procedure TGPU.SetUpdate(Update: boolean);
var
  i: integer;
begin
  for i := 1 to FMaxThreads do
    if (WorkThread[i] <> nil) then
      WorkThread[i].UpdateNeeded := Update;
end;



end.
