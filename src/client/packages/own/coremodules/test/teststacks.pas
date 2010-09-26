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
  private
    stk_ : TStack;
    error_ : TGPUError;
  end; 

implementation

procedure TTestStack.TestBasics;
var b     : TGPUBoolean;
    str   : TGPUString;
    float : TGPUFloat;
begin
  InitStack(stk_);
  pushStr('test', stk_, error_);
  pushFloat(123456, stk_, error_);
  pushBool(true, stk_, error_);
  AssertEquals('Loading things on stack', QUOTE+'test'+QUOTE+', 123456, true', StackToStr(stk_, error_));

  popBool(b, stk_, error_);
  AssertEquals('Popping boolean', true, b);
  popFloat(float, stk_, error_);
  AssertEquals('Popping float', float, 123456);
  popStr(str, stk_, error_);
  AssertEquals('Popping string', 'test', str);



  ReadLn;
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

