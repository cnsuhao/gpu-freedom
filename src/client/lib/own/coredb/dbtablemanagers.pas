unit dbtablemanagers;
{
   TDbTableManager handles all tables in the sqlite database
    specified in the constructor.dbtablemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses clienttables, servertables, channeltables;


type TDbTableManager = class(TObject)
  public
    constructor Create(filename : String);
    destructor Destroy;

    procedure openAll();
    procedure closeAll();

    function getClientTable() : TDbClientTable;
    function getServerTable() : TDbServerTable;
    function getChannelTable() : TDbChannelTable;

    clienttable_    : TDbClientTable;
    servertable_    : TDbServerTable;
    chantable_      : TDbChannelTable;
end;

implementation

constructor TDbTableManager.Create(filename : String);
begin
  servertable_ := TDbServerTable.Create(filename);
  clienttable_ := TDbClientTable.Create(filename);
  chantable_   := TDbChannelTable.Create(filename);
end;


destructor TDbTableManager.Destroy;
begin
 clienttable_.Free;
 servertable_.Free;
 chantable_.Free;
end;

procedure TDbTableManager.openAll();
begin
  clienttable_.Open;
  servertable_.Open;
  chantable_.Open;
end;

procedure TDbTableManager.closeAll();
begin
  clienttable_.Close;
  servertable_.Close;
  chantable_.Close;
end;

function TDbTableManager.getClientTable() : TDbClientTable;
begin
  Result := clienttable_;
end;

function TDbTableManager.getServerTable() : TDbServerTable;
begin
  Result := servertable_;
end;

function TDbTableManager.getChannelTable() : TDbChannelTable;
begin
  Result := chantable_;
end;


end.
