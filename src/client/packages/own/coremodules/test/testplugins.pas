unit testplugins;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  plugins;

type

  TTestPlugin= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestHookUp; 
  end; 

implementation

procedure TTestPlugin.TestHookUp; 
begin
  Fail('Write your own test'); 
end; 

procedure TTestPlugin.SetUp; 
begin

end; 

procedure TTestPlugin.TearDown; 
begin

end; 

initialization

  RegisterTest(TTestPlugin); 
end.

