unit testargretrievers; 

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  argretrievers;

type

  TTestArgRetriever= class(TTestCase)
  published
    procedure TestHookUp; 
  end; 

implementation

procedure TTestArgRetriever.TestHookUp; 
begin
end;



initialization

  RegisterTest(TTestArgRetriever); 
end.

