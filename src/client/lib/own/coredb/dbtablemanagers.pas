unit dbtablemanagers;
{
   TDbTableManager handles all tables in the sqlite database
    specified in the constructor.dbtablemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses nodetable;


type TDbTableManager = class(TObject)
  public
    constructor Create(filename : String);
    destructor Destroy;

    procedure openAll();
    procedure closeAll();

    function getNodeTable() : TDbNodeTable;

    nodetable_ : TDbNodeTable;
end;

implementation

constructor TDbTableManager.Create(filename : String);
begin
  nodetable_ := TDbNodeTable.Create(filename);
end;


destructor TDbTableManager.Destroy;
begin
 nodetable_.Free;
end;

procedure TDbTableManager.openAll();
begin
  nodetable_.Open;
end;

procedure TDbTableManager.closeAll();
begin
  nodetable_.Close;
end;

function TDbTableManager.getNodeTable() : TDbNodeTable;
begin
  Result := nodetable_;
end;

end.
