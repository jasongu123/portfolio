import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.regex.Pattern;




public class main {
	
	static ArrayList<Person> persons = new ArrayList<Person>();
	static ArrayList<Person> personst = new ArrayList<Person>();
	static String stopwords="about , all along also although among and any anyone anything are around because been before being both but came come coming could did each else every for from get getting going got gotten had has have having her here hers him his how however into its like may most next now only our out particular same she should some take taken taking than that the then there these they this those throughout too took very was went what when which while who why will with without would yes yet you your com doc edu encyclopedia fact facts free home htm html http information internet net new news official page pages resource resources pdf site sites usa web wikipedia www one ones two three four five six seven eight nine ten tens eleven twelve dozen dozens thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred hundreds thousand thousands million millions";
	static ArrayList<Term> totalterms=new ArrayList<Term>();
	static ArrayList<Title> titles=new ArrayList<Title>();
	static ArrayList<Probability> probabilities=new ArrayList<Probability>();
	static ArrayList<Possibilities> posst=new ArrayList<Possibilities>();
	
	public static void main(String[] args) throws Exception {
		read();
		probability();
		int correct=0;
		for(Person person:persons) {
			int index=0;
			System.out.println(person.name);
			while(person.terms[index]!=null) {
				System.out.println(person.terms[index]);
				index++;
			}
		}
		System.out.println(" ");
		for(Term eterm:totalterms) {
			System.out.print(eterm.term+" ");
			System.out.println(eterm.times);
		}
		System.out.println(" ");
		for(Title ptitle : titles) {
			System.out.print(ptitle.title+" ");
			System.out.print(ptitle.times+" ");
			System.out.println((float)ptitle.frequency);
		}
		probabilities();
		System.out.println(" ");
		for(Probability p : probabilities) {
			System.out.println("("+p.word+"|"+p.title+") = "+p.frequency);
		}
		String[] guesses=probabilities2();
		
		for(int k=0;k<personst.size();k++) {
			float[] probs=posst.get(k).logs;
			float minimum=probs[0];
			int mindex=0;
			float totalprob=0;
			for(int i=0;i<probs.length;i++) {
				if(probs[i]<minimum) {
					minimum=probs[i];
					mindex=i;
				}
			}
			for(int i=0;i<probs.length;i++) {
				probs[i]=(float) Math.pow(2,(minimum-probs[i]));
			}
			for(int i=0;i<probs.length;i++) {
				totalprob+=probs[i];
			}
			for(int i=0;i<probs.length;i++) {
				probs[i]=(float)probs[i]/totalprob;
			}
			System.out.print(personst.get(k).name+".   "+"Prediction: "+guesses[k]+"   ");
			if(guesses[k].equals(personst.get(k).title)) {
				System.out.println("Correct.");
				correct++;
			}
			else {
				System.out.println("Incorrect.");
			}
			for(int i=0;i<probs.length;i++) {
				System.out.print(posst.get(k).titles[i]+": ");
				System.out.print(probs[i]+"   ");
			}
			System.out.println(" ");
		}
		System.out.println("Overall accuracy: "+correct+" out of "+posst.size()+" = "+(float)correct/(float)posst.size());
	}
	static void read() throws Exception{	
		System.out.print("Enter path to input file:\n");
		Scanner scan = new Scanner(System.in);
		String path = scan.next();
		System.out.print("Enter number of entries:\n");
		Scanner scan3=new Scanner(System.in);
		int entryNum=Integer.parseInt(scan3.next());
		File in = new File(path);
		Scanner scan2 = new Scanner(in);
		String line = null;
		String name=null;
		String title=null;
		
		Person a;
		int personsNum=0;
		int lines=0;
		int termsNum=0;
		while(scan2.hasNextLine()&&personsNum<entryNum)
		{
			line=scan2.nextLine();
				termsNum=0;
				lines++;
				if(lines==1) {
					name=line;
				}
				if(lines==2) {
					line=line.replaceAll(" ", "");
					title=line;
					String[] terms=new String[200];
					a=new Person(terms, name, title);
					persons.add(a);
				}
				if(lines>1) {
				line.replaceAll(",", "");
				String[] str = line.split("\\s");
				for(String s : str) {
					s=s.replaceAll(",","");
					s=s.replaceAll("\\.", "");
					s=s.replaceAll(" ", "");
					int repeat=0;
				for(String word : persons.get(personsNum).terms) {
					if((s.equals(word)||stopwords.contains(s.toString())||s.length()<=2)) {
						repeat=1;
					}
				}
				if(repeat==0&&s.length()!=0) {
					persons.get(personsNum).terms[termsNum]=s;
					termsNum++;
				}
				}

		}
				if(!line.equals(null)&&line.equals("")) {
					lines=0;
					personsNum++;
				}
			
	}
		lines=0;
		while(scan2.hasNextLine()) {
			line=scan2.nextLine();
			termsNum=0;
			lines++;
			if(lines==1) {
				name=line;
			}
			if(lines==2) {
				line=line.replaceAll(" ", "");
				title=line;
				String[] terms=new String[200];
				a=new Person(terms, name, title);
				personst.add(a);
				personsNum++;
			}
			if(lines>1) {
			line.replaceAll(",", "");
			String[] str = line.split("\\s");
			for(String s : str) {
				s=s.replaceAll(",","");
				s=s.replaceAll("\\.", "");
				s=s.replaceAll(" ", "");
				int repeat=0;
			for(String word : personst.get(personsNum-1-entryNum).terms) {
				if((s.equals(word)||stopwords.contains(s.toString())||s.length()<=2)) {
					repeat=1;
				}
			}
			if(repeat==0&&s.length()!=0) {
				personst.get(personsNum-1-entryNum).terms[termsNum]=s;
				termsNum++;
			}
			}

	}
			if(!line.equals(null)&&line.equals("")) {
				lines=0;
			}
	
		}
}
static void probability() {
	int ttitles=0;
		for(Person person: persons) {
			int index=0;
			int trepeat=0;
			ttitles++;
			while(person.terms[index]!=null) {
				person.terms[index]=person.terms[index].toLowerCase();
				Term aterm=new Term(person.terms[index], 1);
				int rep=0;
				for(Term sterm : totalterms) {
					if(sterm.term.equals(aterm.term)) {
						sterm.times++;
						rep=1;
						break;
					}
				}
				if(rep==0)
				totalterms.add(aterm);
				index++;
			}
			for(Title atitle : titles) {
				if(person.title.equals(atitle.title)) {
					atitle.times++;
					atitle.frequency=(((float)atitle.times)/(float)ttitles);
					trepeat=1;
					break;
				}
			}
			if(trepeat==0) {
				Title ntitle=new Title(person.title,1,(((float)1)/(float)((titles.size()+1))));
				titles.add(ntitle);
			}
		}
		for(Title atitle : titles) {
			atitle.frequency=(((float)atitle.times)/(float)ttitles);
		}
//			for(Title title:titles) {
//				title.frequency=
//			}
	}
//static void p2() {
//	int ttitles=0;
//		for(Person person: personst) {
//			int index=0;
//			int trepeat=0;
//			ttitles++;
//			while(person.terms[index]!=null) {
//				person.terms[index]=person.terms[index].toLowerCase();
//				Term aterm=new Term(person.terms[index], 1);
//				int rep=0;
//				for(Term sterm : totalterms) {
//					if(sterm.term.equals(aterm.term)) {
//						sterm.times++;
//						rep=1;
//						break;
//					}
//				}
//				if(rep==0)
//				totalterms.add(aterm);
//				index++;
//			}
//			for(Title atitle : titles) {
//				if(person.title.equals(atitle.title)) {
//					atitle.times++;
//					atitle.frequency=(((float)atitle.times)/(float)ttitles);
//					trepeat=1;
//					break;
//				}
//			}
//			if(trepeat==0) {
//				Title ntitle=new Title(person.title,1,(((float)1)/(float)((titles.size()+1))));
//				titles.add(ntitle);
//			}
//		}
//		for(Title atitle : titles) {
//			atitle.frequency=(((float)atitle.times)/(float)ttitles);
//		}
////			for(Title title:titles) {
////				title.frequency=
////			}
//	}
static void probabilities() {
	for(Term term : totalterms) {
		for(Title etitle : titles) {
			int termtitle=0;
		for(Person person:persons) {
			if(person.title.equals(etitle.title)) {
				for(String iterm : person.terms) {
					if(term.term.equals(iterm)) {
						termtitle++;
					}
				}
			}
		}
		float frequency=(float) ((((float)termtitle/(float)etitle.times)+0.1)/(1+(2*0.1)));
		Probability p=new Probability(term.term, etitle.title, frequency);
		probabilities.add(p);
	}
}
}

static String[] probabilities2(){
	String[] titlen=new String[titles.size()];
	String[] names=new String[personst.size()];
	int termnum=0;
	int namepos=0;
	float minimum=0;
	int minindex=0;
	int titleindex=0;
	for(Title a : titles) {
		titlen[titleindex]=a.title;
		titleindex++;
	}
	for(Person person : personst) {
		float[] possibilities=new float[titles.size()];
		termnum=0;
		int pos=0;
		for(Title c : titles) {
			possibilities[pos]=(float) -Math.log((float)(c.frequency+0.1)/(float)(1+(titles.size()*0.1)));
			while(person.terms[termnum]!=null) {
				for(Probability p : probabilities) {
					if(person.terms[termnum].equals(p.word)&&c.title.equals(p.title)) {
				possibilities[pos]+=(float) -Math.log(p.frequency);
					}
					}
				termnum++;
				}
			pos++;
			termnum=0;
		}
		minimum=possibilities[0];
		minindex=0;
		for(int i=0;i<possibilities.length;i++) {
			if(possibilities[i]<minimum) {
				minimum=possibilities[i];
				minindex=i;
			}
		}
		Possibilities b=new Possibilities(person.name, titlen, possibilities);
		posst.add(b);
		names[namepos]=titles.get(minindex).title;
		namepos++;
	}
	return names;
}
}
class Term{
	String term;
	int times;
	
	public Term(String term, int times){
		this.term=term;
		this.times=times;
	}
	
}

class Title{
	String title;
	int times;
	float frequency;
	
	public Title(String title, int times, float frequency){
		this.title=title;
		this.times=times;
		this.frequency=frequency;
	}
	
}

class Probability{
	String word;
	String title;
	float frequency;
	
	public Probability(String word, String title, float frequency) {
		this.word=word;
		this.title=title;
		this.frequency=frequency;
	}
	
}

class Person{
	String name;
	String title;
	String[] terms;
	
	public Person(String[] terms, String name, String title){
		this.name=name;
		this.title=title;
		this.terms=terms;
	}
}

class Possibilities{
	String name;
	String[] titles;
	float[] logs;
	public Possibilities(String name, String[] titles, float[] logs) {
		this.name=name;
		this.titles=titles;
		this.logs=logs;
	}
}
