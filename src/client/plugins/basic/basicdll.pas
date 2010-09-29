unit basicdll;


interface

uses stacks;

function description: TStkString;
function weblinktoplugin: TStkString;
function stkversion: TStkString;

function add(var stk: TStack): boolean;
function sub(var stk: TStack): boolean;
function mul(var stk: TStack): boolean;
function dvd(var stk: TStack): boolean;

function sqrt(var stk: TStack): boolean;
function sqr(var stk: TStack): boolean;
function exp(var stk: TStack): boolean;
function ln(var stk: TStack): boolean;
// trigonometry
function sin(var stk: TStack): boolean;
function cos(var stk: TStack): boolean;
function tan(var stk: TStack): boolean;
function arcsin(var stk: TStack): boolean;
function arccos(var stk: TStack): boolean;
function arctan(var stk: TStack): boolean;

function pow(var stk: TStack): boolean;
function powmod(var stk: TStack): boolean;

function less(var stk: TStack): boolean;
function greater(var stk: TStack): boolean;
function equal(var stk: TStack): boolean;

function trunc(var stk: TStack): boolean;
function round(var stk: TStack): boolean;

function initrnd(var stk: TStack): boolean;
function rnd(var stk: TStack): boolean;
function rndg(var stk: TStack): boolean;

function average(var stk: TStack): boolean;
function total(var stk: TStack): boolean;
function stddev(var stk: TStack): boolean;
function variance(var stk: TStack): boolean;

function pop(var stk: TStack): boolean;

// switches 2 last numbers on the stack
function switch(var stk: TStack): boolean;


implementation


function stkversion : TStkString;
begin
  Result := STACK_VERSION;
end;

function description: TStkString;
begin
  Result := 'Basic.dll contains the most used functions in GPU like ' +
    ' add, sub, mul, dvd, exp, pow, powmod, sqr, sqrt, less, greater, equal, '+
    'trunc, round, pop, average, equal, rnd, rndg, switch, total, stddev, variance... '+
	'Routines in this plugin are intended for simple computations and as support for more complex plugins.';
end;

function weburl: TStkString;
begin
  Result := 'http://gpu.sourceforge.net/virtual.php';
end;


function checkFloatParams(nbParams : Longint; stk : TStack) : Boolean;
var types : TStkTypes;
begin
 if (nbParams<1) or (nbParams>3) then raise Exception.Create('basic: Internal error in checkFloatParams');
 types[1]:=FLOAT_STKTYPE;
 types[2]:=FLOAT_STKTYPE;
 types[3]:=FLOAT_STKTYPE;
 Result  :=typeOfParametersCorrect(nbParams, stk, types);
end;

function retrieveFloatParam(var a : TStkFloat; stk : TStack) : Boolean;
begin
 Result := checkFloatParams(1, stk);
 if Result then popFloat(a, stk) else a := 0;
end;

function retrieveFloatParams(var a : TStkFloat; var b: TStkFloat; stk : TStack) : Boolean;
begin
 Result := checkFloatParams(2, stk);
 if Result then
    begin
      popFloat(b, stk)
      popFloat(a, stk);
    end
 else
    begin
      a := 0;
      b := 0;
    end;
end;

function retrieve3FloatParams(var a : TStkFloat; var b: TStkFloat; var c : TStkFloat; stk : TStack) : Boolean;
begin
 Result := checkFloatParams(3, stk);
 if Result then
    begin
      popFloat(c, stk)
      popFloat(b, stk)
      popFloat(a, stk);
    end
 else
    begin
      a := 0;
      b := 0;
      c := 0;
    end;
end;


function add(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
  Result := retrieveFloatParams(a, b, stk);
  if Result then pushFloat(a+b, stk);
end;


function sub(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
  Result := retrieveFloatParams(a, b, stk);
  if Result then pushFloat(a-b, stk);
end;


function mul(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
  Result := retrieveFloatParams(a, b, stk);
  if Result then pushFloat(a*b, stk);
end;

function dvd(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
  Result := retrieveFloatParams(a, b, stk);
  if Result then pushFloat(a/b, stk);
end;

function sqrt(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.sqrt(a), stk);
end;

function sqr(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.sqr(a), stk);
end;

function exp(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.exp(a), stk);
end;


function ln(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.ln(a), stk);
end;

function sin(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.sin(a), stk);
end;

function cos(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.cos(a), stk);
end;

function tan(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.tan(a), stk);
end;

function arcsin(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.arcsin(a), stk);
end;

function arccos(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.arccos(a), stk);
end;

function arctan(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.arctan(a), stk);
end;

function pow(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
 Result := retrieveFloatParams(a, b, stk);
 if Result then pushFloat(System.Power(a, b), stk);
end;

function powmod(var stk: TStack): boolean;
var basis, exponential, modulo,
    value : TStkFloat;
    i     : Longint;
begin
  Result := retrieve3FloatParams(basis, exponential, modulo, stk);
  if not Result then Exit;

  value := 1;
  for i:=1 to Trunc(exponential) do
    value := value*basis mod modulo;
  Result := pushFloat(value, stk);
end;


function less(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
 Result := retrieveFloatParams(a, b, stk);
 if Result then pushBool(a<b, stk);
end;

function greater(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
 Result := retrieveFloatParams(a, b, stk);
 if Result then pushBool(a>b, stk);
end;

function equal(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
 Result := retrieveFloatParams(a, b, stk);
 if Result then pushBool(a=b, stk);
end;


function trunc(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.trunc(a), stk);
end;


function round(var stk: TStack): boolean;
var a : TStkFloat;
begin
 Result := retrieveFloatParam(a, stk);
 if Result then pushFloat(System.round(a), stk);
end;


function initrnd(var stk: TStack): boolean;
begin
  Randomize;
  Result     := True;
end;

function rnd(var stk: TStack): boolean;
begin
  Result := pushFloat(Random(), stk);
end;

function rndg(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
  Result := retrieveFloatParams(a, b, stk);
  if Result then pushFloat(Math.randG(a, b));
end;


procedure retrieveTotAndAvg(var tot, avg : TStkFloat; var stk : TStack);
var i, count   : Longint;
begin
  tot := 0;
  count := 0;
  for i:=1 to stk.idx do
    if stk.stkType=FLOAT_STKTYPE then
     begin
      tot := tot + getFloat(i, stk);
      Inc(count);
     end;
  if count>0 then avg:=tot/count else avg:=0;
end;


function total(var stk: TStack): boolean;
var tot, avg : TStkFloat;
begin
  retrieveTotAndAvg(tot, avg, stk);
  initStack(stk);
  Result := pushFloat(tot, stk);
end;

function average(var stk: TStack): boolean;
var tot, avg : TStkFloat;
begin
  retrieveTotAndAvg(tot, avg, stk);
  initStack(stk);
  Result := pushFloat(avg, stk);
end;


function variance(var stk: TStack): boolean;
var tot, avg, variance : TStkFloat;
begin
 retrieveTotAndAvg(tot, avg, stk);
 variance := 0;
 for i:=1 to stk.idx do
   if stk.stkType=FLOAT_STKTYPE then
      variance := variance + System.sqr(getFloat(i, stk)-avg);

  initStack(stk);
  Result := pushFloat(variance, stk);
end;

function stddev(var stk: TStack): boolean;
begin
  Result := variance(stk);
  Stk.Stack[1] := System.sqrt(Stk.Stack[1]);
end;


function pop(var stk: TStack): boolean;
begin
 Result := mvIdx(stk, -1);
end;

function switch(var stk: TStack): boolean;
var a, b : TStkFloat;
begin
 Result := retrieveFloatParams(a, b, stk);
 if Result then
      begin
        pushFloat(b);
        pushFloat(a);
      end;
end;

end.
