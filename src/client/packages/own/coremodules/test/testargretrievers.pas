unit testargretrievers; 

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  argretrievers, gpuconstants, stacks;

type

  TTestArgRetriever= class(TTestCase)
  published
    procedure TestBasics;
    procedure TestSpaces;
    procedure TestExpressions;
    procedure TestRecursiveExprs;
  protected
    procedure SetUp; override;
    procedure TearDown; override;

  private
    procedure checkArguments(argRetr : TArgRetriever);
  end; 

implementation

procedure TTestArgRetriever.checkArguments(argRetr : TArgRetriever);
var
    arg     : TArgStk;
    error   : TStkError;
begin
  clearError(error);
  arg := argRetr.getArgument(error);
  AssertEquals('First argument is float type', STK_ARG_FLOAT, arg.argtype);
  AssertEquals('First argument value', 5.345, arg.argvalue);

  arg := argRetr.getArgument(error);
  AssertEquals('Second argument is boolean type', STK_ARG_BOOLEAN, arg.argtype);
  AssertEquals('Second argument value', true, (arg.argvalue>0));

  arg := argRetr.getArgument(error);
  AssertEquals('Third argument is boolean type', STK_ARG_BOOLEAN, arg.argtype);
  AssertEquals('Third argument value', false, (arg.argvalue>0));

  arg := argRetr.getArgument(error);
  AssertEquals('Fourth argument is string type', STK_ARG_STRING, arg.argtype);
  AssertEquals('Fourth argument value', 'test', arg.argstring);

  arg := argRetr.getArgument(error);
  AssertEquals('Fifth argument is call type', STK_ARG_CALL, arg.argtype);
  AssertEquals('Fifth argument value', 'plugcall', arg.argstring);

  arg := argRetr.getArgument(error);
  AssertEquals('Sixth argument is empty string', STK_ARG_STRING, arg.argtype);
  AssertEquals('Sixth argument value', '', arg.argstring);
end;

procedure TTestArgRetriever.TestBasics;
var argRetr : TArgRetriever;
begin
  argRetr := TArgRetriever.Create('5.345,true,false,'+QUOTE+'test'+QUOTE+',plugcall,'+QUOTE+QUOTE);
  checkArguments(argRetr);
  argRetr.Free;
end;


procedure TTestArgRetriever.TestSpaces;
var argRetr : TArgRetriever;
begin
  argRetr := TArgRetriever.Create('   5.345    ,true, false   ,    '+QUOTE+'test'+QUOTE+'  ,    plugcall   ,  '+QUOTE+QUOTE+'   ,');
  checkArguments(argRetr);
  argRetr.Free;
end;

procedure TTestArgRetriever.TestExpressions;
var
    arg     : TArgStk;
    error   : TStkError;
    argRetr : TArgRetriever;
begin
  clearError(error);
  argRetr := TArgRetriever.Create(' ( 1,  1,  add), {3,  2,   mul} ');
  arg := argRetr.getArgument(error);
  AssertEquals('First argument is expression type', STK_ARG_EXPRESSION, arg.argtype);
  AssertEquals('First argument value', ' 1,  1,  add', arg.argstring);

  arg := argRetr.getArgument(error);
  AssertEquals('Second argument is expression type', STK_ARG_EXPRESSION, arg.argtype);
  AssertEquals('Second argument value', '3,  2,   mul', arg.argstring);
  argRetr.Free;
end;

procedure TTestArgRetriever.TestRecursiveExprs;
var
    arg     : TArgStk;
    error   : TStkError;
    argRetr : TArgRetriever;
begin
  clearError(error);
  argRetr := TArgRetriever.Create(' ( 1,  1, (2, 3, mul), add), { ( 3,  2) , {1,2,3,(2,3) },   mul} ');
  arg := argRetr.getArgument(error);
  AssertEquals('First argument is expression type', STK_ARG_EXPRESSION, arg.argtype);
  AssertEquals('First argument value', ' 1,  1, (2, 3, mul), add', arg.argstring);

  arg := argRetr.getArgument(error);
  AssertEquals('Second argument is expression type', STK_ARG_EXPRESSION, arg.argtype);
  AssertEquals('Second argument value', ' ( 3,  2) , {1,2,3,(2,3) },   mul', arg.argstring);
  argRetr.Free;

end;


procedure TTestArgRetriever.SetUp;
begin
end;

procedure TTestArgRetriever.TearDown;
begin
end;


initialization

  RegisterTest(TTestArgRetriever); 
end.

