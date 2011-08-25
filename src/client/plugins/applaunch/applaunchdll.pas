unit applaunchdll;

{$mode objfpc}{$H+}

interface

uses SysUtils, stacks, formatsets;

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

function launch_wait(var stk: TStack): boolean;
begin
end;

function launch_nowait(var stk: TStack): boolean;
begin
end;


end.

