unit teststacks;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  stacks;

type

  TTestStack= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestHookUp;
  private
    stk_ : TStack;
    error_ : TGPUError;
  end; 

implementation

procedure TTestStack.TestHookUp; 
begin
  pushStr('test', stk_, error_);
  pushFloat(123456, stk_, error_);
  pushBool(true, stk_, error_);

  WriteLn(StackToStr(stk_, error_));
  ReadLn;
end; 

procedure TTestStack.SetUp; 
begin
  InitStack(stk_);
end; 

procedure TTestStack.TearDown; 
begin

end; 

initialization

  RegisterTest(TTestStack); 
end.

