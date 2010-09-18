unit basicdll;


interface

uses Common, Definitions, Math;

function description: PChar;
function weblinktoplugin: PChar;

function add(var stk: TStack): boolean; stdcall;
function sub(var stk: TStack): boolean; stdcall;
function mul(var stk: TStack): boolean; stdcall;
function dvd(var stk: TStack): boolean; stdcall;

function sqrt(var stk: TStack): boolean; stdcall;
function sqr(var stk: TStack): boolean; stdcall;
function exp(var stk: TStack): boolean; stdcall;
function ln(var stk: TStack): boolean; stdcall;
// trigonometry
function sin(var stk: TStack): boolean; stdcall;
function cos(var stk: TStack): boolean; stdcall;
function tan(var stk: TStack): boolean; stdcall;
function arcsin(var stk: TStack): boolean; stdcall;
function arccos(var stk: TStack): boolean; stdcall;
function arctan(var stk: TStack): boolean; stdcall;

function pow(var stk: TStack): boolean; stdcall;
function powmod(var stk: TStack): boolean; stdcall;

function less(var stk: TStack): boolean; stdcall;
function greater(var stk: TStack): boolean; stdcall;
function equal(var stk: TStack): boolean; stdcall;

function trunc(var stk: TStack): boolean; stdcall;
function round(var stk: TStack): boolean; stdcall;

function initrnd(var stk: TStack): boolean; stdcall;
function rnd(var stk: TStack): boolean; stdcall;
function rndg(var stk: TStack): boolean; stdcall;

function average(var stk: TStack): boolean; stdcall;
function total(var stk: TStack): boolean; stdcall;
function stddev(var stk: TStack): boolean; stdcall;
function variance(var stk: TStack): boolean; stdcall;

function pop(var stk: TStack): boolean; stdcall;

// switches 2 last numbers on the stack
function switch(var stk: TStack): boolean; stdcall;


implementation

var
  RandomSeed: longint;

 {function DisplayMsg(s:String):Boolean;stdcall;
 begin
  ShowMessage(s);
  Result := True;
 end;}

function description: PChar;
begin
  Result := PChar('Basic.dll contains the most used functions in GPU like ' +
    ' add, sub, mul, dvd, exp, pow, powmod, sqr, sqrt, less, greater, equal, '+
    'trunc, round, pop, average, equal, rnd, rndg, switch, total, stddev, variance... '+
	'Routines in this plugin are intended for simple computations and as support for more complex plugins.');
	
	
end;

function weblinktoplugin: PChar;
begin
  Result := PChar('http://gpu.sourceforge.net/virtual.php');
end;

function add(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx; {this should speed up and also made the code more readable}

  {check if enough parameter}
  if Idx < 2 then
    Exit;

  Stk.Stack[Idx - 1] := Stk.Stack[Idx - 1] + Stk.Stack[Idx];

  {never forget to set the Idx right at the end}
  stk.StIdx := Idx - 1;
  Result    := True;
end;


function sub(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then
    Exit;

  Stk.Stack[Idx - 1] := Stk.Stack[Idx - 1] - Stk.Stack[Idx];

  stk.StIdx := Idx - 1;
  Result    := True;
end;


function mul(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then
    Exit;

  Stk.Stack[Idx - 1] := Stk.Stack[Idx - 1] * Stk.Stack[Idx];

  stk.StIdx := Idx - 1;
  Result    := True;
end;

function dvd(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then
    Exit;

  Stk.Stack[Idx - 1] := Stk.Stack[Idx - 1] / Stk.Stack[Idx];

  stk.StIdx := Idx - 1;
  Result    := True;
end;

function sqrt(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;

  Stk.Stack[Idx] := System.sqrt(Stk.Stack[Idx]);

  Result := True;
end;

function sqr(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;

  Stk.Stack[Idx] := System.sqr(Stk.Stack[Idx]);

  Result := True;
end;

function exp(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;

  Stk.Stack[Idx] := System.exp(Stk.Stack[Idx]);

  Result := True;
end;


function ln(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;

  Stk.Stack[Idx] := System.ln(Stk.Stack[Idx]);

  Result := True;
end;

function sin(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;
  Stk.Stack[Idx] := System.sin(Stk.Stack[Idx]);
  Result := True;
end;

function cos(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;
  Stk.Stack[Idx] := System.cos(Stk.Stack[Idx]);
  Result := True;
end;

function tan(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;
  Stk.Stack[Idx] := Math.tan(Stk.Stack[Idx]);
  Result := True;
end;

function arcsin(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;
  Stk.Stack[Idx] := Math.arcsin(Stk.Stack[Idx]);
  Result := True;
end;

function arccos(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;
  Stk.Stack[Idx] := Math.ArcCos(Stk.Stack[Idx]);
  Result := True;
end;

function arctan(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;
  Stk.Stack[Idx] := System.ArcTan(Stk.Stack[Idx]);
  Result := True;
end;

function pow(var stk: TStack): boolean; stdcall;
var
  Idx, Count: integer;
  temp: extended;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then
    Exit;

  temp := 1;
  for Count := 1 to System.Trunc(Stk.Stack[Idx]) do
  begin
    temp := temp * Stk.Stack[Idx - 1];
  end;

  Stk.Stack[Idx - 1] := temp;
  stk.StIdx := Idx - 1;
  Result    := True;
end;

function powmod(var stk: TStack): boolean; stdcall;
var
  Idx:    integer;
  Count:  integer;
  temp:   extended;
  temp2:  integer;
  Modulo: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 3 then
    Exit;

  Modulo := System.Trunc(Stk.Stack[Idx]);
  temp2  := 1;
  for Count := 1 to System.Trunc(Stk.Stack[Idx - 1]) do
  begin
    temp2 := temp2 * System.Trunc(Stk.Stack[Idx - 2]) mod Modulo;
  end;
  Stk.Stack[Idx - 2] := temp2;

  stk.StIdx := Idx - 2;
  Result    := True;
end;


function less(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then
    Exit;

  if (Stk.Stack[Idx - 1] < Stk.Stack[Stk.StIdx]) then
    Stk.Stack[Idx - 1] := 1  {first arg is less than second arg}
  else
    Stk.Stack[Idx - 1] := 0;
  {first arg is equal or more than second arg}

  stk.StIdx := Idx - 1;
  Result    := True;
end;

function greater(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then
    Exit;

  if (Stk.Stack[Idx - 1] > Stk.Stack[Stk.StIdx]) then
    Stk.Stack[Idx - 1] := 1  {first arg is less than second arg}
  else
    Stk.Stack[Idx - 1] := 0;
  {first arg is equal or more than second arg}

  stk.StIdx := Idx - 1;
  Result    := True;
end;

function equal(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 2 then
    Exit;

  if (Stk.Stack[Idx - 1] = Stk.Stack[Stk.StIdx]) then
    Stk.Stack[Idx - 1] := 1  {first arg is less than second arg}
  else
    Stk.Stack[Idx - 1] := 0;
  {first arg is equal or more than second arg}

  stk.StIdx := Idx - 1;
  Result    := True;
end;


function trunc(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;

  stk.Stack[Idx] := System.Trunc(stk.Stack[Idx]);

  Result := True;
end;


function round(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then
    Exit;

  stk.Stack[Idx] := System.Round(stk.Stack[Idx]);

  Result := True;
end;


function initrnd(var stk: TStack): boolean; stdcall;
begin
  Randomize;
  RandomSeed := RandSeed;
  Result     := True;
end;

function rnd(var stk: TStack): boolean; stdcall;
begin
  Result := False;
  if Stk.StIdx < MAXSTACK then
    Inc(Stk.StIdx)
  else
    Exit;
  RandSeed   := RandomSeed;
  Stk.Stack[Stk.StIdx] := Random;
  RandomSeed := RandSeed;
  Result     := True;
end;

function rndg(var stk: TStack): boolean; stdcall;
begin
  Result := False;
  if Stk.StIdx < 2 then
    Exit;
  RandSeed   := RandomSeed;
  Stk.Stack[Stk.StIdx - 1] := Math.RandG(Stk.Stack[Stk.StIdx - 1], Stk.Stack[Stk.StIdx]);
  Stk.StIdx  := Stk.StIdx - 1;
  RandomSeed := RandSeed;
  Result     := True;
end;


function total(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
  i  : longint;
  total: Double;
begin
  Result := False;
  Idx    := stk.StIdx;
  if Idx < 1 then Exit;

  total := 0;
  For i := 1 to Idx  do
  	total := total + Stk.Stack[i];
  Stk.Stack[1] := total;
  Stk.StIdx := 1;
  {never forget to set the Idx right at the end}
  Result    := True;
end;

function average(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
  Idx    := stk.StIdx;
  Result := total(stk);
  Stk.Stack[1] := Stk.Stack[1] / Idx;
end;


function variance(var stk: TStack): boolean; stdcall;
var
  Idx, i  : longint;
  total, average: Extended;
begin
  Result := False;
  Idx    := stk.StIdx; {this should speed up and also made the code more readable}

  {check if enough parameter}
  if Idx < 1 then
    Exit;

  total := 0;
  For i := 1 to Idx  do
  	total := total + Stk.Stack[i];
  average := total/idx;
  total := 0;
  For i := 1 to Idx  do
  	total := total + System.sqr(Stk.Stack[i]-average);


  Stk.Stack[1] := total/Idx;
  Stk.StIdx := 1;
  {never forget to set the Idx right at the end}
  Result    := True;
end;

function stddev(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
  i  : longint;
  total, average: Extended;
begin
  Result := variance(stk);
  Stk.Stack[1] := System.sqrt(Stk.Stack[1]);
end;


function pop(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
begin
 if Stk.StIdx > 0 then Stk.StIdx := Stk.StIdx - 1;
 Result := True;
end;

function switch(var stk: TStack): boolean; stdcall;
var
  Idx: integer;
  tmpExtended : Extended;
  tmpPChar : PChar;
begin
 Result := False;
 if Stk.StIdx < 2 then
    Exit;
 
 // switch numbers 
 tmpExtended := Stk.Stack[Stk.StIdx ];
 Stk.Stack[Stk.StIdx ] := Stk.Stack[Stk.StIdx - 1];
 Stk.Stack[Stk.StIdx - 1] := tmpExtended;

 // switch PChars
 tmpPChar := Stk.PCharStack[Stk.StIdx ];
 Stk.PCharStack[Stk.StIdx ] := Stk.PCharStack[Stk.StIdx - 1];
 Stk.PCharStack[Stk.StIdx - 1] := tmpPChar;
 Result := True;
end;

end.
