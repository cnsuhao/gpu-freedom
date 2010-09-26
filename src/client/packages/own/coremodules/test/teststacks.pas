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
  end; 

implementation

procedure TTestStack.TestHookUp; 
begin
  Fail('Write your own test'); 
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

