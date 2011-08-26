unit applaunchdll;

{$mode objfpc}{$H+}

interface

uses SysUtils, stacks, formatsets, Process;

function description    : TStkString;
function weburl         : TStkString;
function stkversion     : TStkString;

function launch_wait(var stk: TStack): boolean;
function launch_nowait(var stk: TStack): boolean;

implementation

function description: TStkString;
begin
  Result := 'applaunch.dll contains functions to launch external executables: ' +
    'launch_wait and launch_nowait.';
end;

function weburl: TStkString;
begin
  Result := 'http://gpu.sourceforge.net';
end;

function stkversion: TStkString;
begin
  Result := STACK_VERSION;
end;

function launch(var stk: TStack; wait : Boolean) : boolean;
var a,b : TStkString;
    aProcess : TProcess;
begin
 Result := retrieveStringParams(a, b, stk);
 if not Result then Exit;

 Result := false;
 AProcess := TProcess.Create(nil);
 try
    AProcess.CommandLine := '"'+a+'" "'+b+'"';
    if wait then
         AProcess.Options := AProcess.Options + [poWaitOnExit]
    else
         AProcess.Options := AProcess.Options - [poWaitOnExit];

    AProcess.Execute;
    Result := true;
 finally
    AProcess.Free;
 end;
end;

function launch_wait(var stk: TStack): boolean;
begin
  Result := launch(stk, true);
end;

function launch_nowait(var stk: TStack): boolean;
begin
  Result := launch(stk, false);
end;


end.

