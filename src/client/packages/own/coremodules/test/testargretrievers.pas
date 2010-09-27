unit testargretrievers; 

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  argretrievers;

type

  TTestArgRetriever= class(TTestCase)
  published
    procedure TestBasics;
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  end; 

implementation

procedure TTestArgRetriever.TestBasics;
begin
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

