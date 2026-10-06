/**
 * Questions Clef answers as an offline teacher. They paraphrase each dataset's own label definitions
 * and were fixed before any result was seen; changing them changes the cache key, so answers to old
 * wording are never mixed with new.
 */
import type { Questions } from './clef.ts';

export const TOXICCHAT_QUESTIONS: Questions = {
  toxic: { type: 'noul', instructions: 'Is this message, written by a user to an AI chatbot, toxic? Toxic means it is offensive, hateful, harassing, sexually explicit, violent, or requests or tries to get the chatbot to produce such content or to help with harmful or illegal activity.' },
  jailbreak: { type: 'noul', instructions: 'Is this message, written by a user to an AI chatbot, a jailbreaking attempt? Jailbreaking means trying to trick or manipulate the chatbot into producing content it should refuse, for example through role play, pretend scenarios, hypothetical framing, or instructions to ignore its rules.' },
};

/**
 * Banking77's heads are groups of intents (src/data.ts INTENTS): urgent = possible fraud or loss
 * (compromised or lost/stolen card or phone, a card payment, cash withdrawal or direct debit the
 * customer doesn't recognise); refund = asking for a refund, or a refund not showing up.
 */
export const BANKING77_QUESTIONS: Questions = {
  urgent: { type: 'noul', instructions: 'Is this message, from a bank customer to customer support, reporting a possible security or fraud problem? That means their card or details may be compromised, their card or phone is lost or stolen, or they see a card payment, cash withdrawal or direct debit on their account that they do not recognise.' },
  refund: { type: 'noul', instructions: 'Is this message, from a bank customer to customer support, about a refund? That means they want to request a refund for a purchase, or a refund they expected has not shown up in their account.' },
};
