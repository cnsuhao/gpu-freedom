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

    procedure TestFloats;
    procedure TestBooleans;
    procedure TestStrings;
    procedure TestIndexes;


  private
    stk_   : TStack;
  end;

implementation

procedure TTestStack.TestBasics;
var b     : TStkBoolean;
    str   : TStkString;
    float : TStkFloat;
    types : TStkTypes;
begin
  InitStack(stk_);
  AssertEquals('Stack index should be 0', 0, stk_.idx);

  pushStr('test', stk_);
  pushFloat(123456, stk_);
  pushBool(true, stk_);
  AssertEquals('Loading things on stack', QUOTE+'test'+QUOTE+', 123456, true', StackToStr(stk_));

  AssertEquals('Stack index should be 3', 3, stk_.idx);
  AssertEquals('First param should be string', true, isStkString(1, stk_));
  AssertEquals('Second param should be float', true, isStkFloat(2, stk_));
  AssertEquals('Third param should be boolean', true, isStkBoolean(3, stk_));

  AssertEquals('First param shouldnt be boolean', false, isStkBoolean(1, stk_));
  AssertEquals('Second param shouldnt be string', false, isStkString(2, stk_));
  AssertEquals('Third param shouldnt be float', false, isStkFloat(3, stk_));

  AssertEquals('Three parameters are enough', true, enoughParametersOnStack(3, stk_));
  AssertEquals('Four parameters are not enough', false, enoughParametersOnStack(4, stk_));

  types[1] := STRING_STKTYPE;
  types[2] := FLOAT_STKTYPE;
  types[3] := BOOLEAN_STKTYPE;
  AssertEquals('Type of parameters correct in call with 3 args', true, typeOfParametersCorrect(3, stk_, types));

  // assuming a call which needs only two arguments
  types[1] := FLOAT_STKTYPE;
  types[2] := BOOLEAN_STKTYPE;
  types[3] := NO_STKTYPE;
  AssertEquals('Type of parameters correct in call with 2 args', true, typeOfParametersCorrect(2, stk_, types));

  // assuming a call which needs only one argument
  types[1] := BOOLEAN_STKTYPE;
  types[2] := NO_STKTYPE;
  types[3] := NO_STKTYPE;
  AssertEquals('Type of parameters correct in call with 1 arg', true, typeOfParametersCorrect(1, stk_, types));

  // assuming a call which does not need arguments
  types[1] := NO_STKTYPE;
  AssertEquals('Type of parameters correct in call without args', true, typeOfParametersCorrect(0, stk_, types));

  types[1] := FLOAT_STKTYPE;
  AssertEquals('Type of parameters has to be incorrect', false, typeOfParametersCorrect(1, stk_, types));

  str := getStr(1, stk_);
  AssertEquals('Content of string has to be', 'test', str);
  float := getFloat(2, stk_);
  AssertEquals('Content of string has to be', 123456, float);
  b := getBool(3, stk_);
  AssertEquals('Content of b has to be', true, b);

  popBool(b, stk_);
  AssertEquals('Popping boolean', true, b);
  popFloat(float, stk_);
  AssertEquals('Popping float', float, 123456);
  popStr(str, stk_);
  AssertEquals('Popping string', 'test', str);

  AssertEquals('Stack should be empty now, but...', true, isEmptyStack(stk_));
end; 

procedure TTestStack.TestOverrun;
var i : Longint;
begin
  InitStack(stk_);
  for i:=1 to MAX_STACK_PARAMS do
    pushFloat(i, stk_);

  AssertEquals('Max stack is not reached', false, maxStackReached(stk_));

  pushStr('Overrun', stk_);
  AssertEquals('Overrun error', TOO_MANY_ARGUMENTS_ID, stk_.error.errorId);
  AssertEquals('Stk.idx stays at ', MAX_STACK_PARAMS, stk_.idx);
  clearError(stk_.error);
  AssertEquals('No error', NO_ERROR_ID, stk_.error.errorId);

  pushFloat(123456, stk_);
  AssertEquals('Overrun error', TOO_MANY_ARGUMENTS_ID, stk_.error.errorId);
  AssertEquals('Stk.idx stays at ', MAX_STACK_PARAMS, stk_.idx);
  clearError(stk_.error);
  AssertEquals('No error', NO_ERROR_ID, stk_.error.errorId);

  pushBool(false, stk_);
  AssertEquals('Overrun error', TOO_MANY_ARGUMENTS_ID, stk_.error.errorId);
  AssertEquals('Stk.idx stays at ', MAX_STACK_PARAMS, stk_.idx);
  clearError(stk_.error);
  AssertEquals('No error', NO_ERROR_ID, stk_.error.errorId);
end;

procedure TTestStack.TestFloats;
begin
end;

procedure TTestStack.TestBooleans;
begin
end;

procedure TTestStack.TestStrings;
begin
end;

procedure TTestStack.TestIndexes;
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

