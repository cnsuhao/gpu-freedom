unit testmethodcontrollers; 

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  methodcontrollers;

type

  TTestMethodController= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestMethodController;
  end; 

implementation

procedure TTestMethodController.TestMethodController;
var meth : TMethodController;
begin
 meth := TMethodController.Create();
 meth.allowRunningFunctionConcurrently('terragen');
 meth.allowRunningFunctionConcurrently('p2psearch');
 meth.registerMethodCall('add', 'basic', 1);
 meth.registerMethodCall('concat', 'strbasic', 2);
 AssertEquals('Method call on slot 1 is', 'add', meth.getMethodCall(1));
 AssertEquals('Method call on slot 2 is', 'concat', meth.getMethodCall(2));
 AssertEquals('Plugin name on slot 1 is', 'basic', meth.getPluginName(1));
 AssertEquals('Plugin name on slot 2 is', 'strbasic', meth.getPluginName(2));
 AssertEquals('sjflksjf is not called', false, meth.isAlreadyCalled('sjflksjf'));
 AssertEquals('add is called', true, meth.isAlreadyCalled('add'));
 AssertEquals('concat is called', true, meth.isAlreadyCalled('concat'));


 // allowed concurrently
 meth.registerMethodCall('terragen', 'earthsim', 3);
 meth.registerMethodCall('p2psearch', 'searchengine', 4);
 AssertEquals('terragen is not called', false, meth.isAlreadyCalled('terragen'));
 AssertEquals('p2psearch is not called', false, meth.isAlreadyCalled('p2psearch'));
 AssertEquals('Method call on slot 3 is', 'terragen', meth.getMethodCall(3));
 AssertEquals('Method call on slot 4 is', 'p2psearch', meth.getMethodCall(4));
 AssertEquals('Plugin name on slot 3 is', 'earthsim', meth.getPluginName(3));
 AssertEquals('Plugin name on slot 4 is', 'searchengine', meth.getPluginName(4));

 meth.unregisterMethodCall(1);
 AssertEquals('add is not called', false, meth.isAlreadyCalled('add'));
 AssertEquals('concat is called', true, meth.isAlreadyCalled('concat'));
 meth.clear();
 AssertEquals('concat is not called', false, meth.isAlreadyCalled('concat'));


 meth.Free;
end; 

procedure TTestMethodController.SetUp; 
begin

end; 

procedure TTestMethodController.TearDown; 
begin

end; 

initialization

  RegisterTest(TTestMethodController); 
end.

