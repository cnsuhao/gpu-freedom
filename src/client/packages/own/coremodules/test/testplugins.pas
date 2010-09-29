unit testplugins;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  plugins, stacks;

type

  TTestPlugin= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestBasicPlugin;
  end; 

implementation

procedure TTestPlugin.TestBasicPlugin;
var plugin : TPlugin;
    appPath : String;
    stk     : TStack;
begin
 AppPath := ExtractFilePath(ParamStr(0));
 plugin  := TPlugin.Create(AppPath+PathDelim+'plugins','basic','dll');
 AssertEquals('Plugin is not loaded', false, plugin.isloaded());
 AssertEquals('Loading plugin', true, plugin.load());
 AssertEquals('Plugin is loaded', true, plugin.isloaded());
 AssertEquals('Method add exists', true, plugin.method_exists('add'));
 AssertEquals('Method asdfsadfsdf does not exist', false, plugin.method_exists('asdfsadfsdf'));
 AssertEquals('Method stkversion exists', true, plugin.method_exists('stkversion'));
 AssertEquals('Stkversion matches', STACK_VERSION, plugin.getDescription('stkversion'));

 clearStk(stk);
 pushFloat(1, stk);
 pushFloat(1, stk);
 AssertEquals('Call to add', true, plugin.method_execute('add', stk));
 AssertEquals('1+1 is', 2, getFloat(1, stk));

 clearStk(stk);
 pushFloat(49, stk);
 AssertEquals('Call to sqrt', true, plugin.method_execute('sqrt', stk));
 AssertEquals('Square root of 49 is', 7, getFloat(1, stk));

 clearStk(stk);
 pushFloat(11, stk);
 AssertEquals('Call to sqr', true, plugin.method_execute('sqr', stk));
 AssertEquals('11 squared is', 121, getFloat(1, stk));


 clearStk(stk);
 pushFloat(5, stk);
 pushFloat(3, stk);
 AssertEquals('Call to dvd', true, plugin.method_execute('dvd', stk));
 AssertEquals('5/3 is', 5/3, getFloat(1, stk));


 AssertEquals('Discarding plugin', true, plugin.discard());
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

