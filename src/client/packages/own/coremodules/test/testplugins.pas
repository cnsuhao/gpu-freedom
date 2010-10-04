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
    procedure TestStrBasicPlugin;
  end; 

implementation

procedure TTestPlugin.TestBasicPlugin;
var plugin : TPlugin;
    appPath : String;
    stk     : TStack;

    piMonte : TStkFloat;
begin
 AppPath := ExtractFilePath(ParamStr(0));
 plugin  := TPlugin.Create(AppPath+PathDelim+'plugins/lib','basic','dll');
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

 clearStk(stk);
 pushFloat(1E6, stk);
 AssertEquals('Call to montecarlo_pi', true, plugin.method_execute('montecarlo_pi', stk));
 popFloat(piMonte, stk);
 AssertEquals('Pi between 3.0 and 3.2', true, (piMonte>=3) and (piMonte<=3.2));

 AssertEquals('Discarding plugin', true, plugin.discard());
end;


procedure  TTestPlugin.TestStrBasicPlugin;
var AppPath : String;
    plugin  : TPlugin;
    stk     : TStack;
    str,
    str2    : TStkString;
    a       : TStkFloat;
    i       : Longint;
begin
 AppPath := ExtractFilePath(ParamStr(0));
 plugin  := TPlugin.Create(AppPath+PathDelim+'plugins/lib','strbasic','dll');
 AssertEquals('Plugin is not loaded', false, plugin.isloaded());
 AssertEquals('Loading plugin', true, plugin.load());
 AssertEquals('Plugin is loaded', true, plugin.isloaded());
 AssertEquals('Method concat exists', true, plugin.method_exists('concat'));
 AssertEquals('Method asdfsadfsdf does not exist', false, plugin.method_exists('asdfsadfsdf'));
 AssertEquals('Method stkversion exists', true, plugin.method_exists('stkversion'));
 AssertEquals('Stkversion matches', STACK_VERSION, plugin.getDescription('stkversion'));

 clearStk(stk);
 pushStr('freedom ', stk);
 pushStr('light my fire', stk);
 AssertEquals('Call to concat', true, plugin.method_execute('concat', stk));
 popStr(str, stk);
 AssertEquals('Concat is', 'freedom light my fire', str);

 clearStk(stk);
 pushStr('cadabra', stk);
 pushStr('abracadabra', stk);
 AssertEquals('Call substr', true, plugin.method_execute('substr', stk));
 popFloat(a, stk);
 AssertEquals('Pos of cadabra is', 5, a);

 clearStk(stk);
 pushStr('computational ', stk);
 pushStr('gpu is revolution', stk);
 pushFloat(8, stk);
 AssertEquals('Call to insert', true, plugin.method_execute('insert', stk));
 popStr(str, stk);
 AssertEquals('Insert is', 'gpu is computational revolution', str);

 clearStk(stk);
 pushStr('gpu is not good', stk);
 pushFloat(8, stk);
 pushFloat(4, stk);
 AssertEquals('Call to delete', true, plugin.method_execute('delete', stk));
 popStr(str, stk);
 AssertEquals('Delete is', 'gpu is good', str);


 // testing a huge string
 clearStk(stk);
 str  := '';
 str2 := '';
 pushStr(str, stk);
 for i:=1 to 1000 do
      begin

      end;

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

