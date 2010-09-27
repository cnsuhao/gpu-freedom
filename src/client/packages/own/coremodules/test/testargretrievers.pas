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
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  end; 

implementation

procedure TTestArgRetriever.TestBasics;
var argRetr : TArgRetriever;
    arg     : TArgStk;
    error   : TStkError;
begin
  clearError(error);
  argRetr := TArgRetriever.Create('5.345,true,false,'+QUOTE+'test'+QUOTE+',plugcall');

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


  argRetr.Free;
end;


procedure TTestArgRetriever.TestSpaces;
var argRetr : TArgRetriever;
    arg     : TArgStk;
    error   : TStkError;
begin

  argRetr := TArgRetriever.Create('   5.345    ,true, false   ,    '+QUOTE+'test'+QUOTE+'  ,    plugcall   ');

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

