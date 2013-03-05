program gpucore;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp,
  { you can add units after this }
  coreloops;

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
begin
 if coreloop_.getCoreMonitor().coreCanStart then
    coreloop_.start()
 else
    begin
         WriteLn('Could not launch core, as it is already running. '+#13#10+
                 'Please delete the lockfile locks/coreapp.lock to stop the process and before relaunching it.');
         ReadLn;
         Halt;
    end;

  while coreloop_.getCoreMonitor().coreCanRun do
    begin
     coreloop_.tick();
     Sleep(1000);
    end;
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

  mainLoop;

  // stop program loop
  Terminate;
end;


constructor TGPUCoreApp.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
  StopOnException:=True;

  coreloop_ := TCoreLoop.Create();
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

