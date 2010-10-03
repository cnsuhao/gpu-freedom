unit testfrontendmanagers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  frontendmanagers;

type

  TTestFrontendManager= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestFrontendManager;
  private
    frontman_ : TFrontendManager;
  end; 

implementation

procedure TTestFrontendManager.TestFrontendManager;
begin
end;

procedure TTestFrontendManager.SetUp; 
begin
  frontman_ := TFrontendManager.Create();
end; 

procedure TTestFrontendManager.TearDown; 
begin
  frontman_.Free;
end; 

initialization

  RegisterTest(TTestFrontendManager); 
end.

