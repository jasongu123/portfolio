import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;

public class front 
{
	int numSteps;//max number of steps to complete the maze
	int atomCount = 0;//number of atoms
	ArrayList<Atom> atomList = new ArrayList<Atom>();//a list of all of the atoms
	ArrayList<ArrayList<Atom>> clauses = new ArrayList<ArrayList<Atom>>();//a list of the clauses
	ArrayList<ArrayList<Integer>> intClauses = new ArrayList<ArrayList<Integer>>();//the list of clauses as numerical literals
	Node[] nodes;//list of the nodes of the maze
	String[] nodeNames;//the node names
	String[] treasureNames;//the treasure names

	public static void main(String[] args) throws Exception
	{
		front puzzle = new front();
		puzzle.read();
		puzzle.initClauses();
		puzzle.initIntClauses();
		puzzle.write();
	}
	
	//read the input file
	void read() throws Exception
	{
		System.out.print("Enter path to input file:\n");
		Scanner kbd = new Scanner(System.in);
		String path = kbd.next();
		BufferedReader in = new BufferedReader(new FileReader(path));
		nodeNames = in.readLine().split("\\s+");
		//create and set names of all of the nodes
		nodes = new Node[nodeNames.length];
		for(int i = 0; i < nodeNames.length; i++)
		{
			nodes[i] = new Node(nodeNames[i]);
		}
		treasureNames = in.readLine().split("\\s+");
		numSteps = Integer.parseInt(in.readLine());
		String line = null;
		int i = 0;
		while ((line = in.readLine()) != null) 
		{	
			//read in the appropriate elements of each node
			String[] elements = line.split("\\s+");
			int j = 2;
			while(!elements[j].equals("NEXT"))
			{
				Treasure t = new Treasure(elements[j]);
				nodes[i].treasures.add(t);
				j++;
			}
			j++;
			while(j < elements.length)
			{
				Node n = null;
				for(int k = 0; k < nodes.length; k++)
				{
					if((nodes[k].name).equals(elements[j]))
					{
						n = nodes[k];
						break;
					}
				}
				nodes[i].next.add(n);
				j++;
			}
			i++;
		}
		in.close();
	}
	
	//initialize all of the propositional clauses
	void initClauses()
	{
		//1st proposition: (You can only be at one node at one time period.)
		//1st
		for(int time = 0; time <= numSteps; time++)
		{
			for(int i = 0; i < nodes.length; i++)
			{
				String node = nodeNames[i];
				At x = new At(node, time);
				x.negative = true;
				atomList.add(x);
				for(int j = i+1; j < nodes.length; j++)
				{
					node = nodeNames[j];
					At y = new At(node, time);
					y.negative = true;
					clauses.add(new ArrayList<Atom>());
					clauses.get(clauses.size()-1).add(x);
					clauses.get(clauses.size()-1).add(y);
				}
			}
		}
		
		
		//2nd set of propositions (The player must move on edges.)
		for(int time = 0; time < numSteps; time++)
		{
			for(int i = 0; i < nodes.length; i++)
			{
				At x = new At(nodeNames[i], time);
				x.negative = true;
				clauses.add(new ArrayList<Atom>());
				clauses.get(clauses.size()-1).add(x);
				for(int j = 0; j < nodes[i].next.size(); j++)
				{
					At y = new At(nodes[i].next.get(j).name, time + 1);
					clauses.get(clauses.size()-1).add(y);
				}
			}
		}
		
		
		//3rd set of propositions (If the player is on a certain spot that contains the treasure at a certain time, the player will get the treasure at that time.) 
		for(int time = 0; time <= numSteps; time++)
		{
			for(int i = 0; i < nodes.length; i++)
			{
				for(int j = 0; j < nodes[i].treasures.size(); j++)
				{

					At y = new At(nodeNames[i], time);
					y.negative = true;
					Has z = new Has(nodes[i].treasures.get(j).name, time);
					atomList.add(z);
					clauses.add(new ArrayList<Atom>());
					clauses.get(clauses.size() - 1).add(y);
					clauses.get(clauses.size() - 1).add(z);
				}
			}
		}
		

		
		
		//4th: If a treasure is in possession at time T, it will be in possession at time T+1.
		for(int time = 0; time < numSteps; time++)
		{
			for(int i = 0; i < nodes.length; i++)
			{
				for(int j = 0; j < nodes[i].treasures.size(); j++)
				{
					Has x = new Has(nodes[i].treasures.get(j).name, time);
					x.negative = true;
					atomList.add(x);
					for(int k = 0; k < nodes.length; k++)
					{
						if(nodes[k] == nodes[i])
						{
							Has z = new Has(nodes[i].treasures.get(j).name, time + 1);
							atomList.add(z);
							clauses.add(new ArrayList<Atom>());
							clauses.get(clauses.size() - 1).add(x);
							clauses.get(clauses.size() - 1).add(z);
						}
					}
				}
			}
		}
		
		//5th set of propositions (Let M1 ... Mq be the nodes that supply treasure T. If the player does not have treasure T at time I-1 and has T at time I, then at time I they must be at one of the nodes M1 ... Mq.)
				for(int time = 0; time < numSteps; time++)
				{
					for(int i = 0; i < nodes.length; i++)
					{
						for(int j = 0; j < nodes[i].treasures.size(); j++)
						{
							Has x = new Has(nodes[i].treasures.get(j).name, time);
							//x.negative = true;
							atomList.add(x);
							for(int k = 0; k < nodes.length; k++)
							{
								if(nodes[k] == nodes[i])
								{
									Has y = new Has(nodes[i].treasures.get(j).name, time + 1);
									y.negative = true;
									atomList.add(y);
									At z = new At(nodeNames[k], time + 1);
									atomList.add(z);									
									clauses.add(new ArrayList<Atom>());
									clauses.get(clauses.size() - 1).add(x);
									clauses.get(clauses.size() - 1).add(y);
									clauses.get(clauses.size() - 1).add(z);
								}
							}
						}
					}
				}
		

		
		//6th set of propositions (The player must start at the beginning.)
		At x = new At("START", 0);
		atomList.add(x);
		clauses.add(new ArrayList<Atom>());
		clauses.get(clauses.size() - 1).add(x);
		
		//7th set of propositions (At the beginning, the player has no items.)
		for(int i = 0; i < treasureNames.length; i++)		
		{
			Has y = new Has(treasureNames[i], 0);
			y.negative=true;
			atomList.add(y);
			clauses.add(new ArrayList<Atom>());
			clauses.get(clauses.size() - 1).add(y);
		}
		
		//8th set of propositions (The last step must be the goal.)	
		for(int i = 0; i < treasureNames.length; i++) {
		Has z = new Has(treasureNames[i], numSteps);
		atomList.add(z);
		clauses.add(new ArrayList<Atom>());
		clauses.get(clauses.size() - 1).add(z);
	}
	}
	//convert the propositional atoms into integer literals
	void initIntClauses()
	{
		int count = 1;
		for(int i = 0; i < atomList.size(); i++)
		{
			String atom = atomList.get(i).toString();
			for(int j = 0; j < clauses.size(); j++)
			{
				ArrayList<Atom> line = clauses.get(j);
				for(int k = 0; k < line.size(); k++)
				{
					if(line.get(k).toString().equals(atom))
					{
						if(line.get(k).number == 0) {
							line.get(k).number = count;
						}
					}
				}
			}
			count++;
		}
		
		for(int i = 0; i < clauses.size(); i++)
		{
			intClauses.add(new ArrayList<Integer>());
			ArrayList<Atom> line = clauses.get(i);
			for(int j = 0; j < line.size(); j++)
			{
				int number = 0 - line.get(j).number;
				if(line.get(j).negative == false)
				{
					number = Math.abs(number);
				}
				intClauses.get(i).add(number);
			}
		}
	}
	
	//write the integer literals to a file
	void write() throws Exception
	{
		System.out.print("Enter path to output file:\n");
		Scanner kbd = new Scanner(System.in);
		String path = kbd.next();
		File file = new File(path);
		if(!file.exists())
		{
			file.createNewFile();
		}
		
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		int max = 0;
		for(int i = 0; i < intClauses.size(); i++)
		{
			for(int j = 0; j < intClauses.get(i).size(); j++)
			{
				bw.write(Integer.toString(intClauses.get(i).get(j)));
				if(Math.abs(intClauses.get(i).get(j)) > max)
				{
					max = Math.abs(intClauses.get(i).get(j));
				}
				if(j < intClauses.get(i).size() - 1)
				{
					bw.write(" ");
				}
			}
			bw.newLine();
		}
		
		bw.write("0");
		bw.newLine();
		
		for(int i = 1; i <= max; i++)
		{
			bw.write(i + " ");
			thisloop:
			for(int j = 0; j < intClauses.size(); j++)
			{
				for(int k = 0; k < intClauses.get(j).size(); k++)
				{
					if(Math.abs(intClauses.get(j).get(k)) == i)
					{
						bw.write(clauses.get(j).get(k).toString());
						break thisloop;
					}
				}
			}
			bw.newLine();
		}
		bw.close();
	}
}

//an object for a propositional atom (3 different subtypes)
class Atom
{
	//fields
	int time;
	boolean negative;
	int number;
}

class At extends Atom
{
	//fields
	String node;
	
	//constructor
	public At(String node, int time)
	{
		this.node = node;
		this.time = time;
		this.negative = false;
		this.number = 0;
	}
	
	@Override
	public String toString()
	{
		String string = "At" + "(" + this.node + "," + this.time + ")";
		return string;
	}
}

class Available extends Atom
{
	//fields
	String treasure;
	
	//constructor
	public Available(String treasure, int time)
	{
		this.treasure = treasure;
		this.time = time;
		this.negative = false;
		this.number = 0;
	}
	
	@Override
	public String toString()
	{
		String string = "Available" + "(" + this.treasure + "," + this.time + ")";
		return string;
	}
}

class Has extends Atom
{
	//fields
	String treasure;
	
	//constructor
	public Has(String treasure, int time)
	{
		this.treasure = treasure;
		this.time = time;
		this.negative = false;
		this.number = 0;
	}
	
	@Override
	public String toString()
	{
		String string = "Has" + "(" + this.treasure + "," + this.time + ")";
		return string;
	}
}

//an object to represent a treasure
class Treasure
{
	//fields
	String name;
	
	//constructor
	public Treasure(String name)
	{
		this.name = name;
	}
}

//an object to represent a node of the maze
class Node
{
	//the fields of each node in the maze
	String name;
	ArrayList<Treasure> treasures;
	ArrayList<Treasure> tolls;
	ArrayList<Node> next;
	
	//constructor
	public Node(String name)
	{
		this.name = name;
		this.treasures = new ArrayList<Treasure>();
		this.tolls = new ArrayList<Treasure>();
		this.next = new ArrayList<Node>();
	}
	
	//constructor
	public Node(String name, ArrayList<Treasure> treasures, ArrayList<Treasure> tolls, ArrayList<Node> next)
	{
		this.name = name;
		this.treasures = treasures;
		this.tolls = tolls;
		this.next = next;
	}
}
