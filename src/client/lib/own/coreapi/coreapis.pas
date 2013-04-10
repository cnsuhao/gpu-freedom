unit coreapis;
{
     TCoreApi is the mother of all APIs which interface to the core client.
     It needs to be initialized with the TDbTableManager as the core client
     interfaces with the outside world through the sqlite database.

     (c) 2002-2013 by HB9TVM and the Global Processing Unit Team
     The source code is under GPL license.
}
interface

uses dbtablemanagers;

type TCoreAPI = class(TObject)
   public
     constructor Create(var tableman : TDbTableManager);
   protected
     tableman_  : TDbTableManager;
end;

implementation


constructor TCoreAPI.Create(var tableman : TDbTableManager);
begin
 tableman_ := tableman;
end;

end.
