import java.util.*;
import java.io.*;


public class DP {
	
	ArrayList<ArrayList<Literal>> clauses = new ArrayList<ArrayList<Literal>>();
	ArrayList<Integer> literals = new ArrayList<Integer>();
	ArrayList<String> backMatter = new ArrayList<String>();
	Boolean[] truths;

	public static void main(String[] args) throws Exception
	{
		DP dp = new DP();
		dp.read();
		dp.truths = new Boolean[dp.literals.size()];
		dp.truths = dp.dp(dp.clauses, dp.truths, dp.literals);
		dp.write();
	}
		
	
	//write the truth value results to the output file
	void write() throws Exception
	{
		System.out.print("Enter path to output file:\n");
		Scanner pscanner = new Scanner(System.in);
		String path = pscanner.next();
		File file = new File(path);
		if(!file.exists())
		{
			file.createNewFile();
		}
		
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		for(int i = 0; i < truths.length; i++)
		{
			if(truths[i] == null)
			{
				bw.write("NO SOLUTION");
				bw.newLine();
				break;
			}
			bw.write((i + 1) + " ");
			if(truths[i])
			{
				bw.write("T");
			}
			else
			{
				bw.write("F");
			}
			bw.newLine();
		}
		bw.write("0");
		bw.newLine();
		for(int i = 0; i < backMatter.size(); i++)
		{
			bw.write(backMatter.get(i));
			bw.newLine();
		}
		bw.close();
	}
	
	//the Davis-Putnam algorithm
	Boolean[] dp(ArrayList<ArrayList<Literal>> clauses, Boolean[] truthValues, ArrayList<Integer> literals)
	{
		//make deep copies of the parameters
		Boolean[] temporary = new Boolean[truthValues.length];
		System.arraycopy(truthValues, 0, temporary, 0, truthValues.length);
		ArrayList<ArrayList<Literal>> ccopies = new ArrayList<ArrayList<Literal>>();
		for(int i = 0; i < clauses.size(); i++)
		{
			ccopies.add(new ArrayList<Literal>());
			for(int j = 0; j < clauses.get(i).size(); j++)
			{
				Literal literal = new Literal(clauses.get(i).get(j));
				ccopies.get(i).add(literal);
			}
		}
		
		//begin with the easy cases
		int easyCase = 1;
		while(easyCase > 0)
		{
			easyCase = 0;
			
			//check if tempClauses is empty
			if(ccopies.size() == 0)
			{
				for(int i = 0; i < temporary.length; i++)
				{
					if(temporary[i] == null)
					{
						temporary[i] = true;
					}
				}
				return temporary;
			}
			
			//check for an empty clause
			for(int i = 0; i < ccopies.size(); i++)
			{
				if(ccopies.get(i).size() == 0)
				{
					return new Boolean[1];
				}
			}
			
			//check for pure literals
			boolean pure;
			Boolean negative;
			for(int i = 0; i < literals.size(); i++)
			{
				pure = true;
				negative = null;
				int val = literals.get(i);
				for(int j = 0; j < ccopies.size(); j++)
				{
					for(int k = 0; k < ccopies.get(j).size(); k++)
					{
						Literal literal = new Literal(ccopies.get(j).get(k));
						if(literal.value == val)
						{
							negative = literal.negative;
							break;
						}
					}
				}
				if(negative != null)
				{
					for(int j = 0; j < ccopies.size(); j++)
					{
						for(int k = 0; k < ccopies.get(j).size(); k++)
						{
							Literal literal = ccopies.get(j).get(k);
							if(literal.value == val)
							{
								if(literal.negative != negative)
								{
									pure = false;
									break;
								}
							}
						}
					}
				}
				else
				{
					pure = false;
				}
				if(pure)
				{
					Literal literal = new Literal(literals.get(i));
					literal.negative = negative;
					if(negative)
					{
						temporary[i] = false;
					}
					else
					{
						temporary[i] = true;
					}
					for(int p = 0; p < ccopies.size(); p++)
					{
						for(int j = 0; j < ccopies.get(p).size(); j++)
						{
							if(ccopies.get(p).get(j).value == literal.value)
							{
								ccopies.remove(p);
								p--;
								break;
							}
						}
					}
					easyCase++;
					break;
				}
			}

			//check for unit clauses
			for(int i = 0; i < ccopies.size(); i++)
			{
				if(ccopies.get(i).size() == 1)
				{
					if(ccopies.get(i).get(0).negative == true)
					{
						temporary[ccopies.get(i).get(0).value - 1] = false;
					}
					else
					{
						temporary[ccopies.get(i).get(0).value - 1] = true;
					}
					Literal literal = new Literal(ccopies.get(i).get(0));
					literal.negative = !temporary[ccopies.get(i).get(0).value - 1];
					ccopies = unitPropagate(literal, ccopies);
					easyCase++;
					break;
				}
			}
		}		
		
		int i;
		for(i = 0; i < ccopies.size(); i++)
		{
			if(ccopies.get(i).size() > 0)
			{
				break;
			}
		}
		Literal literal = new Literal(ccopies.get(i).get(0));
		literal.negative = false;
		temporary[literal.value-1] = true;
		ArrayList<Literal> l = new ArrayList<Literal>();
		l.add(literal);
		ccopies.add(l);
		Boolean[] tempVals = dp(ccopies, temporary, literals);
		if(tempVals[0] != null)
		{
			return tempVals;
		}
		ccopies.remove(ccopies.size() - 1);
		l.remove(l.size() - 1);
		literal.negative = true;
		temporary[literal.value-1] = false;
		l.add(literal);
		ccopies.add(l);
		return dp(ccopies, temporary, literals);
	}
	
	void read() throws Exception
	{
		System.out.print("Enter path to input file:\n");
		Scanner pathscanner = new Scanner(System.in);
		String path = pathscanner.next();
		File file = new File(path);
		Scanner fscanner = new Scanner(file);
		String line = null;
		while(fscanner.hasNextLine())
		{
			line = fscanner.nextLine();
			String[] str = line.split("\\s");
			if(Integer.parseInt(str[0]) == 0)
			{
				while(fscanner.hasNextLine())
				{
					backMatter.add(fscanner.nextLine());
				}
				break;
			}
			ArrayList<Literal> clause = new ArrayList<Literal>();
			for(int i = 0; i < str.length; i++)
			{
				clause.add(new Literal(Integer.parseInt(str[i])));
				literals.add(Math.abs(Integer.parseInt(str[i])));
			}
			clauses.add(clause);
		}
		
		//sort and deduce a list of each unique numerical literal value
		Collections.sort(literals);
		int i = 1;
		int prev = literals.get(0);
		while(i < literals.size())
		{
			while((literals.get(i) == prev))
			{
				literals.remove(i);
				if(i == literals.size())
				{
					break;
				}
			}
			if(i< literals.size())
			{
				prev = literals.get(i);
			}
			i++;
		}
	}
	ArrayList<ArrayList<Literal>> unitPropagate(Literal literal, ArrayList<ArrayList<Literal>> clauses)
	{
		for(int i = 0; i < clauses.size(); i++)
		{
			for(int j = 0; j < clauses.get(i).size(); j++)
			{
				if(clauses.get(i).get(j).value == literal.value)
				{
					if(clauses.get(i).get(j).negative == true)
					{
						if(literal.negative == true)
						{
							clauses.remove(i);
							i--;
							break;
						}
						else
						{
							clauses.get(i).remove(j);
							j--;
						}
					}
					else if(clauses.get(i).get(j).negative == false)
					{
						if(literal.negative == false)
						{
							clauses.remove(i);
							i--;
							break;
						}
						else
						{
							clauses.get(i).remove(j);
							j--;
						}
					}
				}
			}
		}
		return clauses;
	}
}

//an object to represent a literal
class Literal
{
	int value;
	boolean negative;
	
	//constructors
	public Literal(int value)
	{
		this.value = Math.abs(value);
		if(value < 0)
		{
			this.negative = true;
		}
		else
		{
			this.negative = false;
		}
	}
	
	public Literal(Literal literal)
	{
		this.value = literal.value;
		this.negative = literal.negative;
	}
}

