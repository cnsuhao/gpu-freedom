unit teststacks;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  stacks, gpuconstants;

type

  TTestStack= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestBasics;
    procedure TestOverrun;

  private
    stk_ : TStack;
    error_ : TGPUError;
  end; 

implementation

procedure TTestStack.TestBasics;
var b     : TGPUBoolean;
    str   : TGPUString;
    float : TGPUFloat;
    types : TGPUStackTypes;
begin
  InitStack(stk_);
  AssertEquals('Stack index should be 0', 0, stk_.idx);

  pushStr('test', stk_, error_);
  pushFloat(123456, stk_, error_);
  pushBool(true, stk_, error_);
  AssertEquals('Loading things on stack', QUOTE+'test'+QUOTE+', 123456, true', StackToStr(stk_, error_));

  AssertEquals('Stack index should be 3', 3, stk_.idx);
  AssertEquals('First param should be string', true, isGPUString(1, stk_));
  AssertEquals('Second param should be float', true, isGPUFloat(2, stk_));
  AssertEquals('Third param should be boolean', true, isGPUBoolean(3, stk_));

  AssertEquals('First param shouldnt be boolean', false, isGPUBoolean(1, stk_));
  AssertEquals('Second param shouldnt be string', false, isGPUString(2, stk_));
  AssertEquals('Third param shouldnt be float', false, isGPUFloat(3, stk_));

  AssertEquals('Three parameters are enough', true, enoughParametersOnStack(3, stk_, error_));
  AssertEquals('Four parameters are not enough', false, enoughParametersOnStack(4, stk_, error_));

  types[1] := GPU_STRING_STKTYPE;
  types[2] := GPU_FLOAT_STKTYPE;
  types[3] := GPU_BOOLEAN_STKTYPE;
  AssertEquals('Type of parameters correct in call with 3 args', true, typeOfParametersCorrect(3, stk_, types, error_));

  // assuming a call which needs only two arguments
  types[1] := GPU_FLOAT_STKTYPE;
  types[2] := GPU_BOOLEAN_STKTYPE;
  types[3] := GPU_NO_STKTYPE;
  AssertEquals('Type of parameters correct in call with 2 args', true, typeOfParametersCorrect(2, stk_, types, error_));

  // assuming a call which needs only one argument
  types[1] := GPU_BOOLEAN_STKTYPE;
  types[2] := GPU_NO_STKTYPE;
  types[3] := GPU_NO_STKTYPE;
  AssertEquals('Type of parameters correct in call with 1 arg', true, typeOfParametersCorrect(1, stk_, types, error_));

  // assuming a call which does not need arguments
  types[1] := GPU_NO_STKTYPE;
  AssertEquals('Type of parameters correct in call without args', true, typeOfParametersCorrect(0, stk_, types, error_));

  types[1] := GPU_FLOAT_STKTYPE;
  AssertEquals('Type of parameters has to be incorrect', false, typeOfParametersCorrect(1, stk_, types, error_));


  popBool(b, stk_, error_);
  AssertEquals('Popping boolean', true, b);
  popFloat(float, stk_, error_);
  AssertEquals('Popping float', float, 123456);
  popStr(str, stk_, error_);
  AssertEquals('Popping string', 'test', str);

  AssertEquals('Stack should be empty now, but...', true, isEmptyStack(stk_));
end; 

procedure TTestStack.TestOverrun;
begin

end;

procedure TTestStack.SetUp; 
begin
end;

procedure TTestStack.TearDown; 
begin

end; 

initialization

  RegisterTest(TTestStack); 
end.

