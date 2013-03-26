program gpucore;
{$DEFINE DEBUG}
{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp,
  { you can add units after this }
  coreloops, coreobjects, loggers, geoutils, fasttransitionsfromcomputed;

type

  { TGPUCoreApp }

  TGPUCoreApp = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    constructor Create(TheOwner: TComponent); override;
    destructor  Destroy; override;
    procedure   WriteHelp; virtual;
  private
    coreloop_ : TCoreLoop;

    procedure   mainLoop;
  end;

{ TGPUCoreApp }

procedure TGPUCoreApp.mainLoop;
var msg1, msg2 : String;
begin
 coreloop_ := TCoreLoop.Create();
 if coreloop_.getCoreMonitor().coreCanStart then
    coreloop_.start()
 else
    begin
         msg1 := 'TGPUCoreApp> Could not launch core, as lockfile locks/gpucore.lock exists. ';
         msg2 := 'TGPUCoreApp> Please stop any running process named gpucore.exe, delete the mentioned lockfile and relaunch gpucore.exe again.';
         logger.log(LVL_FATAL, msg1); logger.log(LVL_FATAL, msg2);
         {$IFDEF DEBUG}
         WriteLn(msg1); WriteLn(msg2);
         ReadLn;
         {$ENDIF}
         Halt;
    end;

  while coreloop_.getCoreMonitor().coreCanRun do
    begin
     coreloop_.tick();
     Sleep(1000);
    end;

  logger.log(LVL_INFO, 'Normal core shutdown initiated, due to lockfile removal.');
  coreloop_.clearFinishedThreads;
  coreloop_.printThreadManagersStatus;

  while coreloop_.waitingForShutdown do Sleep(300);
  logger.log(LVL_INFO, 'Core was shut down correctly.');
end;

procedure TGPUCoreApp.DoRun;
var
  ErrorMsg : String;
begin
  // quick check parameters
  ErrorMsg:=CheckOptions('h','help');
  if ErrorMsg<>'' then begin
    ShowException(Exception.Create(ErrorMsg));
    Terminate;
    Exit;
  end;

  // parse parameters
  if HasOption('h','help') then begin
    WriteHelp;
    Terminate;
    Exit;
  end;

  Randomize; // init random generator!!!
  mainLoop;

  // stop program loop
  Terminate;
end;


constructor TGPUCoreApp.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
  StopOnException:=True;
end;

destructor TGPUCoreApp.Destroy;
begin
  coreloop_.stop();
  coreloop_.Free;
  inherited Destroy;
end;

procedure TGPUCoreApp.WriteHelp;
begin
  { add your help code here }
  writeln('Usage: ',ExeName,' -h');
end;

var
  Application: TGPUCoreApp;

{$R *.res}

begin
  Application:=TGPUCoreApp.Create(nil);
  Application.Title:='gpucore';
  Application.Run;
  Application.Free;
end.

